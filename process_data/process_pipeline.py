from towers_2_images import towers_2_images
from combine_images import combine_images
from shuffle_sig_bkg import shuffle_sig_bkg
from generate_prep_data import generate_prep_data
from generate_efp import generate_efp
from nnify_efps import nnify_efps
from generate_hl_observables import generate_hl_observables



def process_pipeline():
    # Convert all tower data in root_exports into jet images
    print("RUNNING -> towers_2_images")
    towers_2_images()
    
    # Combine individual jet image files into 1 data file
    print("RUNNING -> combine_images")
    combine_images()
    
    # Combine sig/bkg and shuffle
    print("RUNNING -> shuffle_sig_bkg")
    shuffle_sig_bkg()
    
    # Generate prep data for energyflow batch_compute
    print("RUNNING -> generate_prep_data")
    generate_prep_data()
    
    # Generate HL Jet Substructure Observables from prep_data file
    print("RUNNING -> generate_hl_observables")
    generate_hl_observables()
    
    # Generate EFPs from prep_data
    print("RUNNING -> generate_efp")
    generate_efp()
    
    # nn-ify the EFPs
    print("RUNNING -> nnify_efps")
    nnify_efps()


if __name__ == "__main__":
    process_pipeline()
    