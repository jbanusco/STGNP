import argparse
import itertools
import os


def get_parser():
    default_path='/home/jaume/Desktop/Code/stmgcn/hpc/'
    parser = argparse.ArgumentParser(description="Generate configurations for the project")
    parser.add_argument("--save_folder", type=str, help="Folder to save the configurations", default=default_path)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    save_folder = args.save_folder

    # Define your options
    # normalization_options = ["ZNorm", "Spatial", "NoNorm"]
    normalization_options = ["ZNorm"]
    use_all_options = ["False", "True"]
    # use_similarity_options = ["False", "True"]
    # drop_blood_pools = ["False", "True"]    
    use_similarity_options = ["False"]
    drop_blood_pools = ["True"]
    use_global_data = ["True"]

    # Generate all combinations
    combinations = list(itertools.product(
        normalization_options,
        use_all_options,
        use_similarity_options,
        drop_blood_pools,
        use_global_data,
    ))

    # Define the output file
    output_file = os.path.join(save_folder, "configurations.txt")

    # Write combinations to the file
    with open(output_file, "w") as file:
        for combo in combinations:
            file.write(",".join(combo) + "\n")

    print(f"Configurations written to {output_file}")

if __name__ == "__main__":
    main()