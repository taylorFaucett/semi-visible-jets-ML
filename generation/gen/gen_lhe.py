import os
from dict_replace import dict_replace


def gen_lhe_card():
    # Load the MG5 input template
    mg5_template = f"{proc_path}/lhe_proc/schannel_template.txt"
    d = {
        "nevents": 20000,
        "ebeam1": 6500.0,
        "ebeam2": 6500.0,
        "xqcut": 100,
        "ptj": 50,
        "MY1": 1500,  # Z' mass
        "MXd": 10,  # dark quark mass
    }
    process_file = "%s/lhe_proc/lhe_Zp_%s.txt" % (proc_path, d["MY1"])

    dict_replace(mg5_template, process_file, d)
    print("View file with:")
    print(f"less {process_file}")


def gen_lhe_file():
    print("Using proc_file " + process_file)
    run_cmnd = f"nohup {mg5_path}/bin/mg5_aMC {process_file} > {log_path}/mg5/log.txt"
    print("You can follow the progress with the command:")
    print(f"tail -f {log_path}/mg5/log.txt")
    os.system(run_cmnd)


if __name__ == "__main__":
    gen_lhe_card()
    gen_lhe_file()
