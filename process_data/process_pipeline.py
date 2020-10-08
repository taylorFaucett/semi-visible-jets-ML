from towers_2_images import towers_2_images
from combine_images import combine_images
from shuffle_sig_bkg import shuffle_sig_bkg
from generate_prep_data import generate_prep_data
from generate_efp import generate_efp
from nnify_efps import nnify_efps



def process_pipeline():
    # Convert all tower data in root_exports into jet images
    towers_2_images()
    
    # Combine individual jet image files into 1 data file
    combine_images()
    
    # Combine sig/bkg and shuffle
    shuffle_sig_bkg()
    
    # Generate prep data for energyflow batch_compute
    generate_prep_data()
    
    # Generate EFPs from prep_data
    generate_efp()
    
    # nn-ify the EFPs
    nnify_efps()


if __name__ == "__main__":
    process_pipeline()
    