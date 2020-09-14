#! /usr/bin/python
import os
import sys
import shutil
import fileinput
import glob

# Ideally, you should only need to explicitly define your svj_path and the rest
# of the paths should be made for you from that
home = os.getcwd()
data_path = os.path.join(home, "data")
gen_path = os.path.join(home, "gen")
mg5_path = os.path.join(gen_path, "HEPTools", "MG5")
pythia_path = os.path.join(mg5_path, "HEPTools", "pythia8")
log_path = os.path.join(gen_path, "logs")
set_pythia = "export PYTHIA8=" + pythia_path
DelphesPythia8 = os.path.join(mg5_path, "Delphes", "DelphesPythia8")

# Locations for custom run files and process information
proc_path = os.path.join(gen_path, "process_files")
cmnd_path = os.path.join(proc_path, "cmnd_files")
lhe_path = os.path.join(proc_path, "lhe_proc")

os.chdir(gen_path)


def dict_replace(template_file, output_file, dictionary):
    # Read in the template file
    with open(template_file) as f:
        data = f.read()

    # Make substitutions in the template file text
    for k, v in dictionary.items():
        data = data.replace("$" + str(k), str(v))

    # Save the modified template as a new file
    with open(output_file, "w") as writer:
        writer.write(data)
    return output_file


def swap_pdg_id(run_name):
    lhe_path = os.path.join(
        data_path,
        "raw_data",
        "sig_schannel",
        str(rinv_label),
        str(run_name),
        "Events",
        "run_01",
    )
    lhe_file = os.path.join(lhe_path, "unweighted_events.lhe")
    if not os.path.isfile(lhe_file):
        print("unzipping " + lhe_file + ".gz")
        os.system("gunzip " + lhe_file + ".gz")
    print("replacing PDG ID in " + lhe_file)
    os.system("sed -i 's/5000521/4900101/g' " + lhe_file)


def gen_lhe(run_name):
    # Load the MG5 input template
    mg5_template = os.path.join(
        proc_path, "lhe_proc", "templates", "schannel_template.txt"
    )
    rinv_dir = os.path.join(data_path, "raw_data", "sig_schannel", rinv_label)
    run_dir = os.path.join(rinv_dir, run_name)
    if not os.path.exists(rinv_dir):
        os.mkdir(rinv_dir)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # Define dictionary to replace values in the template file
    d = {
        "run_dir": run_dir,
        "nevents": nEvents,
        "ebeam1": 6500.0,
        "ebeam2": 6500.0,
        "xqcut": 100,
        "ptj": 50,
        "MY1": mY1,  # Z' mass
        "MXd": mXd,  # dark quark mass
    }

    process_file = os.path.join(
        proc_path, "lhe_proc", "sig_schannel_" + run_name + ".txt"
    )
    dict_replace(mg5_template, process_file, d)
    log_file = os.path.join(log_path, "mg5", rinv_label + "-" + run_name + ".log")
    run_cmnd = "nohup " + mg5_path + "/bin/mg5_aMC " + process_file + " > " + log_file
    print(run_cmnd)
    print("You can follow the progress with the command:")
    print("tail -f " + log_file)
    os.system(run_cmnd)

    # Swap pdg-id for pythia run
    print("--- Swapping PDG ID")
    swap_pdg_id(run_name)
    print("--- Finished Swapping PDG ID")


def pythia8_delphes(run_name):
    def gen_dict(nEvents, mY1, mXd, rinv, lambdas, out_path):
        # Defines the dictionary with values to
        # replace in the template file
        d = {
            "nEvents": nEvents,
            "lambdas": lambdas,
            "alpha_fsr": 0.13,
            "pTminFSR": lambdas * 1.1,
            "Z_prime": mY1,
            "dark_quark": mXd,
            "dark_meson": 2.0 * mXd,
            "mWidth": mXd / 50.0,
            "mMin": mXd - mXd / 50.0,
            "mMax": mXd + mXd / 50.0,
            "dark_matter": mXd - 0.01,
            "1_rinv": 1.0 - rinv,
            "rinv": rinv,
            "1_rin_half": round((1 - rinv) / 5.0, 6),
            "output_path": out_path,
        }
        return d

    def cmnd_generator(nEvents, mY1, mXd, rinv, lambdas, out_path):
        # Grab the template cmnd file path
        cmnd_template = os.path.join(cmnd_path, "templates", "rinv_template.cmnd")

        # Define a new output cmnd file (with chosen paramters in the file name)
        new_cmnd = "%s/SVJ_n_%s_mZ_%s_mXd_%s_rinv_%s_lam_%s.cmnd" % (
            cmnd_path,
            nEvents,
            mY1,
            mXd,
            rinv,
            lambdas,
        )

        # Build the dictionary of replacement values
        dictionary = gen_dict(nEvents, mY1, mXd, rinv, lambdas, out_path)

        # Read in the template file
        with open(cmnd_template) as f:
            data = f.read()

        # Make substitutions in the template file text
        for k, v in dictionary.items():
            data = data.replace("$" + str(k), str(v))

        # Save the modified template as a new file
        with open(new_cmnd, "w") as writer:
            writer.write(data)

        return new_cmnd

    def generate_events(cmnd_file, nEvents, mY1, mXd, rinv, lambdas):
        delphes_out_dir = data_path + "/root_files/%s" % (rinv_label)
        if not os.path.exists(delphes_out_dir):
            os.mkdir(delphes_out_dir)
        delphes_out = delphes_out_dir + "/%s.root" % (run_name)
        if os.path.isfile(delphes_out):
            os.remove(delphes_out)

        gen_log_file = log_path + "/delphes/%s-%s.log" % (rinv_label, run_name)
        run_cmnd = "nohup %s %s %s %s > %s" % (
            DelphesPythia8,
            delphes_card,
            cmnd_file,
            delphes_out,
            gen_log_file,
        )
        print(" --- Generating ROOT file with command file")
        print(run_cmnd)
        os.system(run_cmnd)

    delphes_card = os.path.join(
        proc_path, "delphes_cards", "default_cards", "delphes_card_ATLAS.tcl"
    )
    out_path = os.path.join(
        data_path,
        "raw_data",
        "sig_schannel",
        rinv_label,
        run_name,
        "Events",
        "run_01",
        "unweighted_events.lhe",
    )
    cmnd_file = cmnd_generator(nEvents, mY1, mXd, rinv, lambdas, out_path)
    generate_events(cmnd_file, nEvents, mY1, mXd, rinv, lambdas)


if __name__ == "__main__":
    # Dark Parameter Settings
    nEvents = 10000
    mY1 = 1500
    mXd = 10
    lambdas = 5
    rinv = float(sys.argv[2])
    rinv_str = str(rinv).replace(".", "p")
    rinv_label = "rinv-" + rinv_str

    # Set Pythia path for use with Delphes
    os.system(set_pythia)

    # Set the run name (from terminal input)
    print("--- Generating data with rinv and run name:")
    print("     rinv = " + rinv_label)
    run_name = "RUN" + str(sys.argv[1])
    print("--- " + run_name)

    # Generate lhe file with run name
    print("--- Generating LHE file")
    gen_lhe(run_name)
    print("--- LHE file generated")

    # # Run pythia and delphes
    print("--- Running Delphes/Pythia")
    pythia8_delphes(run_name)
