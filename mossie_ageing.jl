using DataArrays, DataFrames, DataFramesMeta
using ScikitLearn
@sk_import preprocessing: StandardScaler


# set pwd
home = homedir()
cd("$home/Documents/•wip/projects/ML_mosquitoes/")

# prep data
df = readtable("mosquitoes_spectra.dat", separator='\t')
df = df[:,2:end]

X = df[:Age]
y = df[2:end]

mapper = DataFrameMapper([, StandardScaler())])

fit_transform!(mapper, copy(y))

# cross validation settings
seed = 4
validation_size = 0.30
num_folds = 10

# pick models



models = []
models.append(("LR", Logis))
