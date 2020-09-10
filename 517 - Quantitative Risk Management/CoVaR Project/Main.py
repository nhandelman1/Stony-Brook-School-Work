import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt

def time_constant_delta_covar_regression(macroData, instData, quantile, instName, writeToFile):
    data = macroData.merge(instData, on='5_Day_Dates')

    mod = smf.quantreg('Fin_Sec_Loss ~ ' + str(instName), data)
    res = mod.fit(q=quantile)
    var50 = data[str(instName)].quantile(0.5)
    varq = data[str(instName)].quantile(quantile)

    if writeToFile:
        f = open(str(instName) + ' Constant Time Delta CoVaR Summary.txt', 'w')
        f.write(str(quantile) + ' Quantile')
        f.write('\n')
        f.write(str(res.summary()))
        f.write('\n')
        f.write('Delta Covar ' + str(quantile) + " : " + str(res.params[1]*(varq - var50)))
        f.close()

def time_varying_delta_covar_regression(macroData, instData, quantile, instName, writeToFile):
    data = macroData.merge(instData, on='5_Day_Dates')

    mod = smf.quantreg(str(instName) + ' ~ Change_3_M_TR + Change_TR_Slope + TED + Baa_3_M_TR + SP_500 + RE_excess_FS + SP_500_Vol', data)
    varqres = mod.fit(q=quantile)
    var50res = mod.fit(q=0.5)
    mod = smf.quantreg('Fin_Sec_Loss ~ Change_3_M_TR + Change_TR_Slope + TED + Baa_3_M_TR + SP_500 + RE_excess_FS + SP_500_Vol + ' + str(instName), data)
    covarqres = mod.fit(q=quantile)

    if writeToFile:
        f = open(str(instName) + ' Time Varying Delta CoVaR Parameters.txt', 'w')
        f.write(str(quantile) + ' Quantile VaR for institution')
        f.write('\n')
        f.write(str(varqres.summary()))
        f.write('\n')
        f.write('\n')
        f.write('0.5 Quantile VaR for institution')
        f.write('\n')
        f.write(str(var50res.summary()))
        f.write('\n')
        f.write('\n')
        f.write(str(quantile) + ' Quantile CoVaR for system given institution')
        f.write('\n')
        f.write(str(covarqres.summary()))
        f.close()

    return {'VaRqParams': varqres.params, 'VaR50Params': var50res.params, 'CoVaRqParams': covarqres.params}

def compute_in_sample_time_varying_delta_covar(macroData, nameParamDict):
    paramNames = ['Intercept',
                  'Change_3_M_TR',
                  'Change_TR_Slope',
                  'TED',
                  'Baa_3_M_TR',
                  'SP_500',
                  'RE_excess_FS',
                  'SP_500_Vol']

    data = macroData.copy()
    data['Intercept'] = 1
    data = data[paramNames]
    tvdcData = pd.DataFrame()

    for name, paramDict in nameParamDict.items():
        varqseries = data.dot(paramDict['VaRqParams'])
        var50series = data.dot(paramDict['VaR50Params'])
        tvdcData[str(name)] = paramDict['CoVaRqParams'].iloc[-1] * (varqseries.subtract(var50series))

    return tvdcData

def plot_in_sample_delta_covar(tvdcData, startDate, endDate):
    tvdcData = tvdcData*100
    ax = tvdcData.plot()
    plt.title('Weekly In-Sample \u0394Covar for 8 Financial Institutions\n' +
              'Over the period ' + str(startDate) + ' - ' + str(endDate))
    ax.set_xlabel("Observations")
    ax.set_ylabel("\u0394Covar %")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()

quantile = 0.99
writeTimeConstantToFile = False
writeTimeVaryingToFile = True

macroData = pd.read_csv("data.csv")
instData = pd.read_csv("institution losses.csv")
instNames = list(instData.columns)[1:]
nameParamDict = {}

for name in instNames:
    time_constant_delta_covar_regression(macroData, instData, quantile, name, writeTimeConstantToFile)
    nameParamDict.update({str(name): time_varying_delta_covar_regression(macroData, instData, quantile, name, writeTimeVaryingToFile)})

tvdcData = compute_in_sample_time_varying_delta_covar(macroData, nameParamDict)
plot_in_sample_delta_covar(tvdcData, macroData['5_Day_Dates'].iloc[0], macroData['5_Day_Dates'].iloc[-1])