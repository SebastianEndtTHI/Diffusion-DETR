import datagen.data_generation_pipeline as gen
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)


    # dataset
    parser.add_argument('--n_samples', default=250000, type=int, help="number of samples to generate.")
    parser.add_argument('--n_comp_list', nargs="+", default=[2, 3, 4, 5], type=int, help="list with possible numbers of compartments in a sample.")


    # diffusion protocol
    parser.add_argument('--diff_file', default="cfin_multib", type=str, help="path to diffusion protocol file, or 'cfin_multib': b <= 1000 protocol of the CFIN in vivo data (Hansen & Jespersen 2016), derived from the dipy download on first use.")


    # compartments
    parser.add_argument('--dist', default="global_uniform", type=str, help="set md/fa value distribution to 'uniform' for uniform sampling within +-3 sd's or 'gm' for normal distribution sampling")

    # noise level
    parser.add_argument('--noise_level', default=0.01, type=float, help="standard deviation of gaussian noise to add to the signal.")
    
    # save path
    parser.add_argument('--save_folder_path', default="datagen/data/train/", type=str)

    # global seed
    parser.add_argument('--seed', default=42, type=int, help="global seed for reproducibility.")

    return parser



def main(args):
    # resolve the default protocol (downloads the CFIN data via dipy on first use)
    if args.diff_file == "cfin_multib":
        from invivo.datasets import cfin_protocol_file
        args.diff_file = cfin_protocol_file()

    # set global seed
    np.random.seed(args.seed)

    # create save folder if it does not exist
    if not os.path.exists(args.save_folder_path):
        os.makedirs(args.save_folder_path)

    # maximum comapartment number in dataset
    max_n_comp = max(args.n_comp_list)
    min_n_comp = min(args.n_comp_list)
    if max_n_comp == min_n_comp:
        comp_setup = "fixed"
    else:
        comp_setup = "dynamic"

    # Ensure equal distribution of number of compartments
    n_samples_per_comp = args.n_samples // len(args.n_comp_list)
    n_comp_array = []
    for c in args.n_comp_list:
        n_comp_array.extend([c] * n_samples_per_comp)
    
    # Pad with random choices if n_samples is not perfectly divisible
    remainder_samples = args.n_samples % len(args.n_comp_list)
    if remainder_samples > 0:
        n_comp_array.extend(np.random.choice(args.n_comp_list, remainder_samples))
        
    np.random.shuffle(n_comp_array)

    # loop generating samples
    dataset = []

    for i in tqdm(range(args.n_samples), miniters=1000):

        # generator initialization
        generator = gen.DTIGenerator(args)
        

        # choose a number of compartments
        n_comp = n_comp_array[i]

        # generate signal and ground truth with n_comp compartments
        S, md, fa, dir, w = generator.generate_Voxel(n_comp)


        # building vector with voxel signal and ground truth for each compartment
        sample = list(S)

        for c in range(n_comp):

            # MD and FA from all compartments in one voxel
            sample.append(md[c])
            sample.append(fa[c])


            # continuous direction as vector with x-, y-, and z-coords
            for coord in dir[c]:
                sample.append(coord)


            # weights from all compartments in one voxel
            sample.append(w[c])


        # padding samples with nan for even size of max_n_comp compartments 
        sample.extend([np.nan] * 6 * (max_n_comp - n_comp))

        # save sample including mri signal (S), repating MD (md), FA (fa), direction (dir), and weights (w) for all compartments, and number of compartments (n_comp)
        sample.append(n_comp)

        dataset.append(sample)


    # signal values defined in columns named 'mri_sigX' with X between 0 and (signal size - 1)
    signal_label = ["mri_sig" + str(s) for s in range(len(S))]


    # compartment metric columns
    base_labels = ["MD", "FA", "dir_X", "dir_Y", "dir_Z", "weight"]


    
    # metric colums named for example 'MD_comp1' describing every metric all compartments per sample
    comp_label = [f"{label}_comp{i}" for i in range(1, max_n_comp + 1) for label in base_labels]

    # last column describes number of compartments per sample
    comp_label.append("n_comp")


    # save dataset in csv file
    column_label = signal_label + comp_label

    df = pd.DataFrame(dataset, columns=column_label)

    # diff dirs number of signal values in diffusion protocol file
    n_diff_dirs = len(generator.direction_table)
    file_prefix = f"comp_setup_{comp_setup}__max_n_comp_{max_n_comp}__min_n_comp_{min_n_comp}__noise_{args.noise_level}__n_signal_values_{n_diff_dirs}"
    df.to_csv(args.save_folder_path + file_prefix + ".csv", index=False)

    # Visualization of total distribution
    try:
        import matplotlib.pyplot as plt
        md_cols = [col for col in df.columns if col.startswith("MD_comp")]
        fa_cols = [col for col in df.columns if col.startswith("FA_comp")]

        all_compartments = []
        for idx in range(len(md_cols)):
            comp_df = df[[md_cols[idx], fa_cols[idx]]].copy()
            comp_df.columns = ["MD", "FA"]
            all_compartments.append(comp_df)

        comp_metrics = pd.concat(all_compartments, ignore_index=True).dropna(subset=["MD", "FA"]) 
        md_vals = comp_metrics["MD"].values
        fa_vals = comp_metrics["FA"].values

        x_min, x_max = 0.0, 4.0
        y_min, y_max = 0.0, 1.0
        nbins_x, nbins_y = 40, 20
        xedges = np.linspace(x_min, x_max, nbins_x + 1)
        yedges = np.linspace(y_min, y_max, nbins_y + 1)

        H, _, _ = np.histogram2d(md_vals, fa_vals, bins=[xedges, yedges])

        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(xedges, yedges)
        pcm = ax.pcolormesh(X, Y, H.T, cmap="magma", vmin=0, shading="auto")
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("Compartment count")
        ax.set_xlabel("Mean diffusivity (MD)")
        ax.set_ylabel("Fractional anisotropy (FA)")
        ax.set_title("Global MD/FA distribution (0-4 MD, 0-1 FA)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig(args.save_folder_path + file_prefix + "_dist.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed, skipping distribution plot.")



# code initialization
if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    # run training
    main(args)
