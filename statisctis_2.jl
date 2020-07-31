# Análisis estadistico en Julia
# Tener una comprensión sólida de las estadísticas en ciencia de datos 
# nos permite comprender mejor nuestros datos y nos permite crear 
# una evaluación cuantificable de cualquier conclusión futura.


# Configurando nuestro manipulador de paquetes
using Pkg;

# Importando los paquetes necesarios
Pkg.add("Statistics")
Pkg.add("StatsBase")
Pkg.add("RDatasets")
Pkg.add("Plots")
Pkg.add("StatsPlots")
Pkg.add("KernelDensity")
Pkg.add("Distributions")
Pkg.add("LinearAlgebra")
Pkg.add("HypothesisTests")
Pkg.add("PyCall")
Pkg.add("MLBase")

# Preprando para usar los paquetes importados
using Statistics
using StatsBase
using RDatasets
using Plots
using StatsPlots
using KernelDensity
using Distributions
using LinearAlgebra
using HypothesisTests
using PyCall
using MLBase


# Obteniendo nuestros datos

D = dataset("datasets", "faithful")
@show names(D)
first(D, 5)

# Obteniendo los resultados de estádistica descriptiva
describe(D)

# Obteniendo los valores de erupciones
eruptions = D[!,:Eruptions]

# Grafica de dispersión de los datos
scatter(
        eruptions,
        label="eruptions"
);

# Obteniendo el tiempo de espera
waittime = D[!,:Waiting]

# Grafica de dispersión de los datos se usa ! para tener como resultado en la misma gráfica
scatter!(
        waittime,
        label="wait time", 
        size=(1000, 300)
);

# Gráficos estadísticos
# Como puede ver, esto no nos dice mucho acerca de los datos ... 
# Probemos algunos gráficos estadísticos

boxplot(
            ["eruption length"],
            eruptions,legend=false,
            size=(700, 300),
            whisker_width=1,
            ylabel="time in minutes"
);

histogram(
            eruptions,
            label="eruptions", 
            size=(800,300)
);

histogram(
            eruptions,bins=:sqrt,
            label="eruptions", 
            size=(400,300)
);

#Estimaciones de densidad del núcleo

#A continuación, veremos cómo podemos ajustar una función de estimación de densidad del núcleo a nuestros datos. Haremos uso del paquete KernelDensity.jl.

p=kde(eruptions)

histogram(eruptions,label="eruptions")
plot!(p.x,p.density .* length(eruptions), linewidth=3,color=2,label="kde fit", size=(400,300)) # nb of elements*bin width

histogram(eruptions,bins=:sqrt,label="eruptions")
plot!(p.x,p.density .* length(eruptions) .*0.2, linewidth=3,color=2,label="kde fit",size=(400,300)) # nb of elements*bin width


myrandomvector = randn(100_000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.1, linewidth=3,color=2,label="kde fit", size=(800,300)) # nb of elements*bin width


d = Normal()
myrandomvector = rand(d,100000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.1, linewidth=3,color=2,label="kde fit", size=(800,300)) # nb of elements*bin width


b = Binomial(40) 
myrandomvector = rand(b,1000000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.5,color=2,label="kde fit", size=(800,300)) # nb of elements*bin width


# A continuación, intentaremos ajustar un conjunto dado de números a una distribución.
x = rand(1000)
d = fit(Normal, x)
myrandomvector = rand(d,1000)
histogram(myrandomvector,nbins=20,fillalpha=0.3,label="fit")
histogram!(x,nbins=20,linecolor = :red,fillalpha=0.3,label="myvector", size=(800,300))


x = eruptions
d = fit(Normal, x)
myrandomvector = rand(d,1000)
histogram(myrandomvector,nbins=20,fillalpha=0.3)
histogram!(x,nbins=20,linecolor = :red,fillalpha=0.3,size=(800,300))


#Hypothesis testing
#A continuación, realizaremos pruebas de hipótesis utilizando el paquete HypothesisTests.jl.

myrandomvector = randn(1000)
OneSampleTTest(myrandomvector)

OneSampleTTest(eruptions)


scipy_stats = pyimport("scipy.stats")
@show scipy_stats.spearmanr(eruptions,waittime)
@show scipy_stats.pearsonr(eruptions,waittime)

scipy_stats.pearsonr(eruptions,waittime)

corspearman(eruptions,waittime)


cor(eruptions,waittime)


scatter(eruptions,waittime,xlabel="eruption length",
    ylabel="wait time between eruptions",legend=false,grid=false,size=(400,300))

#AUC and Confusion matrix

gt = [1, 1, 1, 1, 1, 1, 1, 2]
pred = [1, 1, 2, 2, 1, 1, 1, 1]
C = confusmat(2, gt, pred)   # compute confusion matrix
C ./ sum(C, dims=2)   # normalize per class
sum(diag(C)) / length(gt)  # compute correct rate from confusion matrix
correctrate(gt, pred)
C = confusmat(2, gt, pred)

gt = [1, 1, 1, 1, 1, 1, 1, 0];
pred = [1, 1, 0, 0, 1, 1, 1, 1]
ROC = MLBase.roc(gt,pred)
recall(ROC)
precision(ROC)





