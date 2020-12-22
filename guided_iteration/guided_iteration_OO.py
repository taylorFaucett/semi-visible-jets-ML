import h5py
from average_decision_ordering import calc_ado
from sklearn.metrics import roc_auc_score, roc_curve
import pathlib

path = pathlib.Path.cwd()

class iterationEFP:
    def __init__(self, rinv, iteration, id):
        
        # Iteration details
        self.rinv = rinv
        self.iteration = iteration
        self.id = id
        self.checked = False
        
        # EFP details
        l = self.id.split("_")
        self.n = l[0]
        self.d = l[1]
        self.k = l[2]
        self.kappa = l[4]
        self.beta = l[-1]
        
        # Performance details
        self.auc = None
        self.ado = None
        
    def get_data(self):
        efps_file = path.parent / "data" / "efp" / f"efp-{self.rinv}.h5"
        efps = h5py.File(efps_file, "r")
        X = efps["efps"][self.id][:]
        return X
    
    def get_targets(self):
        efps_file = path.parent / "data" / "efp" / f"efp-{self.rinv}.h5"
        efps = h5py.File(efps_file, "r")
        y = efps["targets"][:]
        return y
    
    def calc_ado(self, gx, ll=False, N=100000):
        fx = self.get_data()
        y = self.get_targets()
        ado_val = calc_ado(fx, gx, y, N)
        if ll:
            self.ado = ado_val
        return ado_val

def guidedIterate(efps):    
    for efp in efps:
        efp
    
def run_guidedIteration(rinv):
    iterations = 10
    for iteration in range(iterations):
        efps = []
        efp_ids = list(h5py.File(path.parent / "data" / "efp" / f"efp-{rinv}.h5", "r")["efps"].keys())
        for efp_id in efp_ids:
            efps.append(guidedIteration(rinv, iteration, efp_id))
        guidedIterate(efps)
    
if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        run_guidedIteration(rinv)
