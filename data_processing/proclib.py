import pandas as pd
import numpy as np
import tqdm
from pyjet import cluster
from pyjet.testdata import get_event


def struc2arr(x):
    # pyjet outputs a structured array. This converts
    # the 4 component structured array into a simple
    # 4xN numpy array
    x = x.view((float, len(x.dtype.names)))
    return x


def jets_2_image(tower, R0, R1, ptmin):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet

    # Convert event to a pyjet compatible format
    # Convert to float64
    tower = tower.astype(np.float64)
    jet_images = pd.DataFrame()
    for entry, event in tower.groupby("entry"):
        trim_pt, trim_eta, trim_phi, trim_mass = [], [], [], []

        # Convert the pandas dataframe to a structured array
        # (pT, eta, phi, mass)
        event = event.to_records(index=False)

        # Convert to a structured array (for pyjet)
        sequence = cluster(event, R=R0, p=-1)
        jets = sequence.inclusive_jets(ptmin=ptmin)

        # Main jets
        jets = sequence.inclusive_jets(ptmin=3)

        # # In case we are missing a leading jet, break early
        # if len(jets) == 0:
        #     return cut_fail()

        # Take just the leading jet
        jet0 = jets[0]

        # Define a cut threshold that the subjets have to meet (i.e. 5% of the original jet pT)
        jet0_max = jet0.pt
        jet0_cut = jet0_max * 0.05

        # Grab the subjets by clustering with R1
        subjets = cluster(jet0.constituents_array(), R=R1, p=1)
        subjet_array = subjets.inclusive_jets()
        for subjet in subjet_array:
            if subjet.pt > jet0_cut:
                # Get the subjets pt, eta, phi constituents
                subjet_data = subjet.constituents_array()
                subjet_data = struc2arr(subjet_data)
                pT = subjet_data[:, 0]
                eta = subjet_data[:, 1]
                phi = subjet_data[:, 2]
                mass = subjet_data[:, 3]

                # Shift all data such that the leading subjet
                # is located at (eta,phi) = (0,0)
                eta -= subjet_array[0].eta
                phi -= subjet_array[0].phi

                # Rotate the jet image such that the second leading
                # jet is located at -pi/2
                # s1x, s1y = subjet_array[1].eta, subjet_array[1].phi
                # theta = np.arctan2(s1y, s1x)
                # if theta < 0.0:
                #     theta += 2 * np.pi
                # eta, phi = rotate(eta, phi, np.pi - theta)

                # Collect the trimmed subjet constituents
                trim_pt.append(pT)
                trim_eta.append(eta)
                trim_phi.append(phi)
                trim_mass.append(mass)

        jet_images = jet_images.append(dfi, ignore_index=True)
        return jet_images
