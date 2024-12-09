from pmf import PMF

def train(input_file):
    # We use a fixed precision for the likelihood.
    # This reflects uncertainty in the dot product.
    # We choose 2 in the footsteps Salakhutdinov
    # Mnihof.
    ALPHA = 2

    # The dimensionality D; the number of latent factors.
    # We can adjust this higher to try to capture more subtle
    # characteristics of each movie. However, the higher it is,
    # the more expensive our inference procedures will be.
    # Specifically, we have D(N + M) latent variables. For our
    # Movielens dataset, this means we have D(2625), so for 5
    # dimensions, we are sampling 13125 latent variables.
    DIM = 10


    pmf = PMF(input_file, DIM, ALPHA, std=0.05)
    pmf.find_map()
    pmf.draw_samples(draws=50, tune=50)

    return pmf
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--output-file', type=str)

    args = parser.parse_args()

    pmf = train(args.input_file)

    import cloudpickle

    with open(args.output_file, "wb") as f:
        cloudpickle.dump(pmf, f)



