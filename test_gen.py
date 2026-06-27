
import numpy as np
import tensorflow as tf

def test_gen():
    X = np.random.randint(0, 100, (1000, 20))
    Y = np.random.randint(0, 100, (1000, 25))
    batch_size = 32
    MAX_ENG = 20
    MAX_SPA = 25
    n = (len(X) // batch_size) * batch_size
    def gen():
        while True:
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx  = perm[start:start + batch_size]
                enc  = X[idx]
                dec  = Y[idx]
                dec_in  = dec[:, :-1]
                dec_out = dec[:, 1:]
                pad_in  = np.zeros((batch_size, MAX_SPA), dtype=np.int32)
                pad_out = np.zeros((batch_size, MAX_SPA), dtype=np.int32)
                sl = dec_in.shape[1]
                pad_in[:, :sl] = dec_in
                pad_out[:, :sl] = dec_out
                yield ({"enc_input": enc, "dec_input": pad_in}, pad_out)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "enc_input": tf.TensorSpec(shape=(batch_size, MAX_ENG), dtype=tf.int32),
                "dec_input": tf.TensorSpec(shape=(batch_size, MAX_SPA), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(batch_size, MAX_SPA), dtype=tf.int32),
        )
    )
    for batch in ds.take(1):
        print("Success yielding batch")
        print("Input keys:", batch[0].keys())
        print("Target shape:", batch[1].shape)

if __name__ == "__main__":
    test_gen()
