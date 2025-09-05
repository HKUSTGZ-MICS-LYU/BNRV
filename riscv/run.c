/* Inference for Ternary Quantized Llama-2 Transformer model (BitNet) in pure C */

#include "sim_stdlib.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "bitnet.h"

#define EXIT_FAILURE 1

extern const char _binary_bin_tokenizer_bin_start[];
extern const char _binary_bin_model_bin_start[];
extern const char _binary_bin_model_bin_end[];
const char* tokenizer_bin = _binary_bin_tokenizer_bin_start;
const char* model_bin = _binary_bin_model_bin_start;

// Profiler Counters
long total_time = 0;
long forward_time = 0;
long rope_time = 0;
long glu_time = 0;
long attention_time = 0;
long ffn_time = 0;
long fmatmul_time = 0;


// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct{
    float* s;
    uint8_t* wq;
} BitNetWeight;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    // float* rms_att_weight; // (layer, dim) rmsnorm weights
    // float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    BitNetWeight* wq; // (layer, dim, n_heads * head_size)
    BitNetWeight* wk; // (layer, dim, n_kv_heads * head_size)
    BitNetWeight* wv; // (layer, dim, n_kv_heads * head_size)
    BitNetWeight* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    BitNetWeight* w1; // (layer, hidden_dim, dim)
    BitNetWeight* w2; // (layer, dim, hidden_dim)
    BitNetWeight* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    return time();
}

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

size_t init_bitnet_weight(BitNetWeight* w, char* ptr, int n_layers, int n, int m){
    char* ptr0 = ptr;
    for (int i = 0; i < n_layers; i++) {
        w[i].s = (float*)ptr;
        ptr += sizeof(float);

        w[i].wq = (uint8_t*)(ptr);
        ptr += n * m * sizeof(uint8_t) / 4; // in case of int2*4 packing
    }
    return ptr - ptr0;
}

void memory_map_weights(TransformerWeights *w, Config* p, char* ptr, int shared_weights){
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;

    w->rms_final_weight = (float*) ptr;
    ptr += p->dim * sizeof(float);

    w->token_embedding_table = (float*)ptr;
    w->wcls = w->token_embedding_table; // shared embedding table
    ptr += p->vocab_size * p->dim * sizeof(float);

    size_t inc = 0;
    w->wq = (BitNetWeight*)malloc(n_layers * sizeof(BitNetWeight));
    inc = init_bitnet_weight(w->wq, ptr, n_layers, p->dim, p->n_heads * head_size);
    ptr += inc;

    w->wk = (BitNetWeight*)malloc(n_layers * sizeof(BitNetWeight));
    inc = init_bitnet_weight(w->wk, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    ptr += inc;

    w->wv = (BitNetWeight*)malloc(n_layers * sizeof(BitNetWeight));
    inc = init_bitnet_weight(w->wv, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    ptr += inc;

    w->wo = (BitNetWeight*)malloc(n_layers * sizeof(BitNetWeight));
    inc = init_bitnet_weight(w->wo, ptr, n_layers, p->n_heads * head_size, p->dim);
    ptr += inc;

    w->w1 = (BitNetWeight*)malloc(n_layers * sizeof(BitNetWeight));
    inc = init_bitnet_weight(w->w1, ptr, n_layers, p->hidden_dim, p->dim);
    ptr += inc;

    w->w2 = (BitNetWeight*)malloc(n_layers * sizeof(BitNetWeight));
    inc = init_bitnet_weight(w->w2, ptr, n_layers, p->dim, p->hidden_dim);
    ptr += inc;

    w->w3 = (BitNetWeight*)malloc(n_layers * sizeof(BitNetWeight));
    inc = init_bitnet_weight(w->w3, ptr, n_layers, p->hidden_dim, p->dim);
    ptr += inc;

    return;
}

void read_checkpoint(Config* config, TransformerWeights* weights) {
    char* ptr = (char*)model_bin;
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number = *(uint32_t*)ptr;
    ptr += sizeof(uint32_t);
    if (magic_number != 0x616b3432) { printf("Bad magic number\n"); exit(EXIT_FAILURE); }
    // read in the version number (uint32), has to be 2
    int version = *(int*)(ptr);
    if (version != 1) { printf( "Bad version %d, need version1\n", version); exit(EXIT_FAILURE); }
    ptr += sizeof(int);
    int header_size = 256; // the header size for version 2 in bytes
    // read in the config header
    if (memcpy(config, ptr, sizeof(Config)) == NULL) { exit(EXIT_FAILURE); }
    ptr += sizeof(Config);
    printf("Model Config:\ndim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
           config->dim, config->hidden_dim, config->n_layers, config->n_heads, config->n_kv_heads, config->vocab_size, config->seq_len);

    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // memory map the Transformer weights into the data pointer
    void* weights_ptr = (char*)(model_bin + header_size); // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}


void build_transformer(Transformer *t) {
    printf("Building Transformer model...\n");
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(&t->config, &t->weights);
    // allocate memory for the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // free the RunState buffers
    free_run_state(&t->state);
}
// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    long start = time_in_ms();
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    rmsnorm_time += time_in_ms() - start;
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void fmatmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    long start = time_in_ms();
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
    fmatmul_time += time_in_ms() - start;
}

void bit_matmul(float* xout, int8_t* qa, BitNetWeight* w, int n, int d, float a_s){
    int32_t *qo = (int32_t*)malloc(d * sizeof(int32_t));

    uint8_t *weight = (uint8_t*)w->wq;
    float s = w->s[0] * a_s;

    qmatmul(qa, qo, weight, n, d);

    long start = time_in_ms();
    // Dequantization
    for(int i = 0; i < d; i++){
        xout[i] = qo[i] * s;
    }
    quant_time += time_in_ms() - start;

    free(qo);
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    size_t qa_size = dim > hidden_dim ? dim : hidden_dim;
    int8_t* qa = (int8_t*)malloc(qa_size * sizeof(int8_t));
    float a_s = 0.0f;
    long start = time_in_ms();
    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        printf("Processing Layer %d/%d\n", (int)l+1, (int)p->n_layers);
        // attention rmsnorm (ignored)
        // rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        bit_rmsnorm(s->xb, x, dim);
        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        a_s = act_scale(s->xb, dim);
        act_quantize(s->xb, qa, a_s, dim);

        // qkv matmuls for this position
        bit_matmul(s->q, qa, w->wq + l, dim, dim, a_s);
        bit_matmul(s->k, qa, w->wk + l, dim, kv_dim, a_s);
        bit_matmul(s->v, qa, w->wv + l, dim, kv_dim, a_s);

        start = time_in_ms();
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        rope_time += time_in_ms() - start;

        // multihead attention. iterate over all heads
        int h;
        start = time_in_ms();
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        attention_time += time_in_ms() - start;

        bit_rmsnorm(s->xb, s->xb, dim);

        a_s = act_scale(s->xb, dim);
        act_quantize(s->xb, qa, a_s, dim);

        // final matmul to get the output of the attention
        bit_matmul(s->xb2, qa, w->wo + l, dim, dim, a_s);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm (ignored)
        // rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);      
        bit_rmsnorm(s->xb, x, dim);

        a_s = act_scale(s->xb, dim);
        act_quantize(s->xb, qa, a_s, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        bit_matmul(s->hb, qa, w->w1 + l, dim, hidden_dim, a_s);
        bit_matmul(s->hb2, qa, w->w3 + l, dim, hidden_dim, a_s);

        start = time_in_ms();
        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        glu_time += time_in_ms() - start;

        // final matmul to get the output of the ffn
        bit_rmsnorm(s->hb, s->hb, hidden_dim);
        
        a_s = act_scale(s->hb, dim);
        act_quantize(s->hb, qa, a_s, dim);

        bit_matmul(s->xb, qa, w->w2 + l, hidden_dim, dim, a_s);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    free(qa);
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    fmatmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    unsigned char* ptr = (unsigned char*)tokenizer_bin;
    // read in the file
    if (!ptr) {
         printf("couldn't load tokenizer\n"); exit(EXIT_FAILURE); 
    }
    if (memcpy(&t->max_token_length, ptr, sizeof(int)) == NULL) {
         printf("failed read\n"); exit(EXIT_FAILURE);
        }
    ptr += sizeof(int);
    // printf("max_token_length=%d\n", t->max_token_length);
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (memcpy(t->vocab_scores + i, ptr, sizeof(float)) == NULL) { printf("failed read\n"); exit(EXIT_FAILURE);}
        ptr += sizeof(float);
        if (memcpy(&len, ptr, sizeof(int) ) == NULL) { printf("failed read\n"); exit(EXIT_FAILURE); }
        ptr += sizeof(int);
        t->vocab[i] = (char *)malloc(len + 1);
        if (memcpy(t->vocab[i], ptr, len) == NULL) { printf("failed read\n"); exit(EXIT_FAILURE); }
        ptr += len;
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (tokscanf(piece, &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex* )bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { printf("cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}


void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        forward_time = time();
        float* logits = forward(transformer, token, pos);
        forward_time = time() - forward_time;

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        printf("Generated:");
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        printf("\n");
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        printf("achieved cycles per tok: %d\n", (end-start)/(pos-1) );
    }

    free(prompt_tokens);
}

int main(){

    float temperature = 0.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 1;            // number of steps to run for
    char *prompt = "";        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length
    printf("Building Tokenizer and Sampler...\n");
    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    
    printf("Generating: \n");
    long start = time_in_ms();
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    total_time += time_in_ms() - start;

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    // Print Profiler Information
    printf("Total Time: %d\n", total_time);
    printf("Forward Time: %d\n", forward_time);
    printf("BitMatmul Time: %d\n", matmul_time);
    printf("RMSNorm Time: %d\n", rmsnorm_time);
    printf("Quant Time: %d\n", quant_time);
    printf("Attention Time: %d\n", attention_time);
    printf("GLU Time: %d\n", glu_time);
    printf("RoPE Time: %d\n", rope_time);
    printf("FMatmul Time: %d\n", fmatmul_time);
    exit(0);
    return 0;
}