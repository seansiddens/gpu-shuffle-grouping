// https://www.pcg-random.org/
// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
static uint pcg_hash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static uint round_function(uint x, uint seed, uint half_index_bits_mask)
{
    // This is a function of both the input, as well as the key
    return (pcg_hash(x ^ seed)) & half_index_bits_mask;
}

static uint encrypt(uint index, uint half_index_bits, uint half_index_bits_mask, uint seed) {
    uint left = (index >> half_index_bits);
    uint right = index & half_index_bits_mask;

    uint new_left = right;
    uint new_right = left ^ round_function(right, seed, half_index_bits_mask);
    left = new_left;
    right = new_right;

    return (left << half_index_bits) | right;
}

static uint decrypt(uint index, uint half_index_bits, uint half_index_bits_mask, uint seed)
{
    uint left = (index >> half_index_bits);
    uint right = index & half_index_bits_mask;

    uint new_right = left;
    uint new_left = right ^ round_function(left, seed, half_index_bits_mask);
    left = new_left;
    right = new_right;

    return (left << half_index_bits) | right;
}

__kernel void shuffle(__global uint *index_bits, 
                      __global uint *seed, 
                      __global uint *output_buf, 
                      __global atomic_uint *success_flag) {
    // Set index bits.
    uint half_index_bits = *index_bits / 2;
    uint half_index_bits_mask = (1 << half_index_bits) - 1;

    // Each thread assigned an index.
    uint index = get_global_id(0);

    // Encryption acts as the bijection beween indices.
    uint shuffled_index = encrypt(index, half_index_bits, half_index_bits_mask, *seed);

    // "undo" the shuffle with a decrypt
    uint unshuffle_index = decrypt(shuffled_index, half_index_bits, half_index_bits_mask, *seed);
    if (index != unshuffle_index) {
        // Something went wrong.
        atomic_store(success_flag, 0);
    }

    output_buf[index] = shuffled_index;
}
