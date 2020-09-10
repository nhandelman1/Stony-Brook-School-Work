import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import csv

#make all vectors numpy arrays
#use decimals for % values (e.g. 52% as 0.52)

##################################################################################################################
##################################################################################################################
##################################################################################################################
# portfolio and portfolio component metrics
def port_ret(vWeight, vRet):
    return vWeight.dot(vRet)

def port_std_dev(vWeight, mCov):
    return np.sqrt(vWeight.dot(mCov).dot(vWeight))

def port_sharpe(vWeight, vMean, mCov):
    return port_ret(vWeight, vMean)/port_std_dev(vWeight, mCov)

def port_prcc(iNumAssets, vWeight, vMean, mCov):
    cprc = port_comp_cprc(vWeight, vMean, mCov)
    return np.sum(np.square(cprc))/iNumAssets

def port_tracking_error(iNumAssets, vWeight, vInitWeight):
    return np.sum(np.square(vWeight-vInitWeight))/iNumAssets

def port_comp_mean(vWeight, vMean):
    return vWeight*vMean

def port_comp_std_dev(vWeight, mCov):
    return vWeight*(mCov.dot(vWeight)/port_std_dev(vWeight, mCov))

def port_comp_cprc(vWeight, vMean, mCov):
    compmean = port_comp_mean(vWeight, vMean)
    compsd = port_comp_std_dev(vWeight, mCov)
    portsharpe = port_sharpe(vWeight, vMean, mCov)
    return compmean - portsharpe*compsd

##################################################################################################################
##################################################################################################################
##################################################################################################################
# return series metrics

def cumu_val(vRets):
    cumuval = 1
    for i in range(vRets.size):
        cumuval *= (vRets[i] + 1)
    return np.round(cumuval, 2)

def geo_ret_ann(vRets):
    return np.round(1200 * (cumu_val(vRets) ** (1/vRets.size) - 1), 2)

def mean_ret_ann(vRets):
    return np.round(1200 * np.mean(vRets), 2)

def std_ret_ann(vRets):
    return np.round(100 * np.sqrt(12) * np.std(vRets), 2)

def sharpe_ret_ann(vRets):
    return np.round(mean_ret_ann(vRets) / std_ret_ann(vRets), 2)

def max_drawdown(vRets):
    prod = 1
    dd = np.array([])

    for i in range(vRets.size):
        prod = np.min(np.array([prod * (1 + vRets[i]), 1]))
        dd = np.append(dd, prod)

    return np.round(100 * (1-np.min(dd)), 2)

def modified_value_at_risk(vRets, nAlpha):
    return np.round(0, 2)


def calc_stats(vRets):
    return [str(cumu_val(vRets)),
            str(geo_ret_ann(vRets)),
            str(mean_ret_ann(vRets)),
            str(std_ret_ann(vRets)),
            str(sharpe_ret_ann(vRets)),
            str(np.round(stats.skew(vRets), 2)),
            str(np.round(stats.kurtosis(vRets), 2)),
            str(max_drawdown(vRets)),
            str(modified_value_at_risk(vRets, 0.95))]

##################################################################################################################
##################################################################################################################
##################################################################################################################
# portfolio optimizations
def inv_vol_port(vSD):
    v = np.reciprocal(vSD)
    return v/np.sum(v)

def equal_weight_port(iNumAssets):
    return np.reciprocal(np.ones(iNumAssets)*iNumAssets)

def min_var_port(iNumAssets, mCov):
    def min_var_obj(vWeight):
        return port_std_dev(vWeight, mCov)

    init = equal_weight_port(iNumAssets)
    cons = ({'type': 'eq', 'fun': budget_cons})
    bnds = bounds(iNumAssets)
    sol = optimize.minimize(min_var_obj, init, bounds=bnds, constraints=cons)
    return sol.x

def equal_risk_port(iNumAssets, mCov):
    def equal_risk_obj(vWeight):
        compsd = port_comp_std_dev(vWeight, mCov)
        sumdiffsq = 0
        for i in range(iNumAssets):
            for j in range(iNumAssets):
                sumdiffsq += (compsd[i] - compsd[j]) ** 2
        return sumdiffsq

    init = equal_weight_port(iNumAssets)
    cons = ({'type': 'eq', 'fun': budget_cons})
    bnds = bounds(iNumAssets)
    sol = optimize.minimize(equal_risk_obj, init, bounds=bnds, constraints=cons)
    return sol.x

def max_diversif_port(iNumAssets, vSD, mCov):
    def max_diversif_obj(vWeight):
        return -1 * np.dot(vWeight,vSD) / port_std_dev(vWeight, mCov)

    init = equal_weight_port(iNumAssets)
    cons = ({'type': 'eq', 'fun': budget_cons})
    bnds = bounds(iNumAssets)
    sol = optimize.minimize(max_diversif_obj, init, bounds=bnds, constraints=cons)
    return sol.x

def max_sharpe_port(iNumAssets, vMean, mCov):
    def max_sharpe_obj(vWeight):
        return -1 * port_ret(vWeight, vMean) / port_std_dev(vWeight, mCov)

    init = equal_weight_port(iNumAssets)
    cons = ({'type': 'eq', 'fun': budget_cons})
    bnds = bounds(iNumAssets)
    sol = optimize.minimize(max_sharpe_obj, init, bounds=bnds, constraints=cons)
    return sol.x

##################################################################################################################
##################################################################################################################
##################################################################################################################
# PRCC optimization

def prcc_optimized_port(iNumAssets, vInitWeight, vMean, mCov, nTECons):
    sharpecons = port_sharpe(vInitWeight, vMean, mCov)

    # port_prcc returns float that is too close to 0
    def prcc_opt_obj(vWeight):
        return 1000000000*port_prcc(iNumAssets, vWeight, vMean, mCov)

    def sharpe_cons(vWeight):
        return port_sharpe(vWeight, vMean, mCov) - sharpecons

    def tracking_error_cons(vWeight):
        return nTECons - port_tracking_error(iNumAssets, vWeight, vInitWeight)

    cons = ({'type': 'eq', 'fun': budget_cons},
            {'type': 'eq', 'fun': sharpe_cons},
            {'type': 'ineq', 'fun': tracking_error_cons})
    bnds = bounds(iNumAssets)
    sol = optimize.minimize(prcc_opt_obj, vInitWeight, method='SLSQP', bounds=bnds, constraints=cons)
    return sol.x

##################################################################################################################
##################################################################################################################
##################################################################################################################
# general optimization constraints
def budget_cons(vWeight):
    return np.sum(vWeight)-1

def bounds(iNumAssets):
    return ((0, np.inf),)*iNumAssets

##################################################################################################################
##################################################################################################################
##################################################################################################################
# calculate portfolios for given data

def calc_portfolios(iNumAssets, mData):
    vMean = np.mean(mData, axis=1)
    vSD = np.std(mData, axis=1)
    mCov = np.cov(mData)

    mRefWeights = np.empty((0, iNumAssets))
    mPRCCWeights = np.empty((0, iNumAssets))

    x = min_var_port(iNumAssets, mCov)
    mRefWeights = np.append(mRefWeights, np.array([x]), axis=0)
    x = prcc_optimized_port(iNumAssets, x, vMean, mCov, 0.1)
    mPRCCWeights = np.append(mPRCCWeights, np.array([x]), axis=0)

    x = inv_vol_port(vSD)
    mRefWeights = np.append(mRefWeights, np.array([x]), axis=0)
    x = prcc_optimized_port(iNumAssets, x, vMean, mCov, 0.1)
    mPRCCWeights = np.append(mPRCCWeights, np.array([x]), axis=0)

    x = equal_weight_port(iNumAssets)
    mRefWeights = np.append(mRefWeights, np.array([x]), axis=0)
    x = prcc_optimized_port(iNumAssets, x, vMean, mCov, 0.1)
    mPRCCWeights = np.append(mPRCCWeights, np.array([x]), axis=0)

    x = equal_risk_port(iNumAssets, mCov)
    mRefWeights = np.append(mRefWeights, np.array([x]), axis=0)
    x = prcc_optimized_port(iNumAssets, x, vMean, mCov, 0.1)
    mPRCCWeights = np.append(mPRCCWeights, np.array([x]), axis=0)

    x = max_diversif_port(iNumAssets, vSD, mCov)
    mRefWeights = np.append(mRefWeights, np.array([x]), axis=0)
    x = prcc_optimized_port(iNumAssets, x, vMean, mCov, 0.1)
    mPRCCWeights = np.append(mPRCCWeights, np.array([x]), axis=0)

    x = max_sharpe_port(iNumAssets, vMean, mCov)
    mRefWeights = np.append(mRefWeights, np.array([x]), axis=0)

    return mRefWeights, mPRCCWeights, vMean, mCov


##################################################################################################################
##################################################################################################################
##################################################################################################################
# write to file
def write_in_sample_data(iNumAssets, lPortNames, mRefWeights, mPRCCWeights, vMean, mCov):
    with open('in sample output.csv','w',newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Traditional Benchmark Portfolios','','','','','','','PRCC Optimized Benchmark Portfolios'])
        writer.writerow(['','Eq-DE','Eq-EM','Bo-Co','NAREIT','Gold','','Eq-DE','Eq-EM','Bo-Co','NAREIT','Gold'])
        writer.writerow([])

        for i in range(iNumAssets):
            #annualized %
            refpmean = np.round(1200*port_ret(mRefWeights[i], vMean), 2)
            refpsd = np.round(100*np.sqrt(12)*port_std_dev(mRefWeights[i], mCov), 2)
            refpsharpe = np.round(refpmean/refpsd, 2)
            refpcompmeans = np.round(1200 * port_comp_mean(mRefWeights[i], vMean), 2)
            refpcompsds = np.round(100*np.sqrt(12) * port_comp_std_dev(mRefWeights[i], mCov), 2)
            refpcprc = np.round(refpcompmeans - (np.sum(refpcompmeans)/np.sum(refpcompsds)) * refpcompsds, 2)
            refpprcc = np.round(np.sum(np.square(refpcprc))/iNumAssets, 2)

            prccpmean = np.round(1200*port_ret(mPRCCWeights[i], vMean), 2)
            prccpsd = np.round(100*np.sqrt(12)*port_std_dev(mPRCCWeights[i], mCov), 2)
            prccpsharpe = np.round(prccpmean/prccpsd, 2)
            prccpte = np.round(100*port_tracking_error(iNumAssets, mRefWeights[i], mPRCCWeights[i]), 2)
            prccpcompmeans = np.round(1200 * port_comp_mean(mPRCCWeights[i], vMean), 2)
            prccpcompsds = np.round(100*np.sqrt(12) * port_comp_std_dev(mPRCCWeights[i], mCov), 2)
            prccpcprc = np.round(prccpcompmeans - (np.sum(prccpcompmeans)/np.sum(prccpcompsds)) * prccpcompsds, 2)
            prccpprcc = np.round(np.sum(np.square(prccpcprc)) / iNumAssets, 2)

            writer.writerow([lPortNames[i]])
            writer.writerow(['','mean='+str(refpmean), 'sd='+str(refpsd), 'sharpe='+str(refpsharpe), 'PRCC='+str(refpprcc),
                             '','','','mean='+str(prccpmean), 'sd='+str(prccpsd), 'sharpe='+str(prccpsharpe),
                             'PRCC='+str(prccpprcc), 'TE='+str(prccpte)])
            writer.writerow(['w*'] + np.round(100*mRefWeights[i], 2).tolist() + [''] + np.round(100*mPRCCWeights[i], 2).tolist())
            writer.writerow(['C mean'] + refpcompmeans.tolist() + [''] + prccpcompmeans.tolist())
            writer.writerow(['C sd'] + refpcompsds.tolist() + [''] + prccpcompsds.tolist())
            writer.writerow(['CPRC'] + refpcprc.tolist() + [''] + prccpcprc.tolist())
            writer.writerow([])


def write_out_sample_data(iNumPorts, lPortNames, mRefRets, mPRCCRets, vMeanTE):
    with open('out sample output.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['', '$', 'GR', 'Mean', 'Sd', 'SR', 'Sk', 'Ku', 'MDD', 'mVaR', 'TE'])
        writer.writerow(['Risk-based portfolios'])

        for i in range(iNumPorts - 1):
            output = calc_stats(mRefRets[i])
            output.insert(0, lPortNames[i])

            writer.writerow(output)

        writer.writerow([])
        writer.writerow(['PRCC-optimized risk-based portfolios'])
        vMeanTE = np.round(100*vMeanTE, 2)

        for i in range(iNumPorts - 1):
            output = calc_stats(mPRCCRets[i])
            output.insert(0, lPortNames[i])
            output.append(str(vMeanTE[i]))

            writer.writerow(output)

        writer.writerow([])
        writer.writerow(['Other benchmark portfolio'])

        output = calc_stats(mRefRets[-1])
        output.insert(0, lPortNames[-1])

        writer.writerow(output)

##################################################################################################################
##################################################################################################################
##################################################################################################################
# run

##################################################################################################################
# load data
lPortNames = ['Minimum Variance Portfolio', 'Inverse Volatility Portfolio', 'Equally Weighted Portfolio',
              'Equal Risk Contribution Portfolio', 'Maximum Diversification Portfolio', 'Maximum Sharpe Portfolio']

# mData each row is asset returns
# remove dates and empty first row
# risk free rate is last column
mData = np.genfromtxt('data combined.csv', delimiter=',')
mData = mData[1:, 1:]
vRiskFree = mData[:, -1]
mData = np.transpose(mData[:, 0:-1])

# convert all returns to returns in excess of risk free rate
mData = np.subtract(mData, vRiskFree)

iNumPorts = len(lPortNames)
iNumAssets = np.size(mData, 0)
iNumObs = np.size(mData, 1)
iRollWindow = 36

##################################################################################################################
# in sample experiment

mRefWeights, mPRCCWeights, vMean, mCov = calc_portfolios(iNumAssets, mData)

write_in_sample_data(iNumAssets, lPortNames, mRefWeights, mPRCCWeights, vMean, mCov)

##################################################################################################################
# out of sample experiment

mRefRets = np.empty((0, iNumPorts))
mPRCCRets = np.empty((0, iNumPorts-1))
vMeanTE = np.zeros(iNumPorts-1)

for i in range(iNumObs-iRollWindow):
    mRefWeights, mPRCCWeights, vMean, mCov = calc_portfolios(iNumAssets, mData[:, i:i+iRollWindow])
    mRefRets = np.append(mRefRets, np.array([mRefWeights.dot(mData[:, i+iRollWindow])]), axis=0)
    mPRCCRets = np.append(mPRCCRets, np.array([mPRCCWeights.dot(mData[:, i+iRollWindow])]), axis=0)

    for j in range(iNumPorts-1):
        vMeanTE[j] += port_tracking_error(iNumAssets, mPRCCWeights[j], mRefWeights[j])

mRefRets = np.transpose(mRefRets)
mPRCCRets = np.transpose(mPRCCRets)
vMeanTE /= iNumObs-iRollWindow

write_out_sample_data(iNumPorts, lPortNames, mRefRets, mPRCCRets, vMeanTE)
