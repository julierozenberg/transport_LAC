import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from pandas import DataFrame

from scipy import interp


def days(length):
    s = InterpolatedUnivariateSpline([0, 5, 30, 50, 200, 300], [0,1*length,5*length,10*length,30*length,50*length], k=1)
    return s

def steps(cm,redirection):
    if redirection==0:
        if cm<15:
            percent = 0
        elif (cm>=15)&(cm<25):
            percent = 0.3
        elif (cm>=25)&(cm<35):
            percent = 0.5
        elif (cm>=35)&(cm<60):
            percent = 0.7
        else:
            percent = 1
    elif redirection==1:
        if cm<10:
            percent = 0
        elif (cm>=10)&(cm<20):
            percent = 0.4
        elif (cm>=20)&(cm<30):
            percent = 0.6
        elif (cm>=30)&(cm<45):
            percent = 0.8
        else:
            percent = 1
    elif redirection==2:
        if cm<5:
            percent = 0
        elif (cm>=5)&(cm<10):
            percent = 0.5
        elif (cm>=10)&(cm<15):
            percent = 0.7
        elif (cm>=15)&(cm<30):
            percent = 0.9
        else:
            percent = 1
    return percent

def annual_losses(expected_losses,losses_col):
    '''
    Calculates expected annual losses based on losses per return period event.
    '''
    cost=expected_losses.copy()
    #keeps the last two points for extrapolation
    #sorts rows by return period
    cost.sort(columns='RP', inplace=True)
    cost.index=range(len(cost))
    cost['frequency']=1/cost['RP']
    #calculates the integral using the trapeze method
    inte=0
    for i in range(1,len(cost)):
        trapeze =(cost.loc[i-1,'frequency']-cost.loc[i,'frequency'])*(cost.loc[i,losses_col]+cost.loc[i-1,losses_col])/2
        inte += trapeze
    inte += cost.loc[len(cost)-1,losses_col]*cost.loc[len(cost)-1,'frequency']
    return inte
	
def totcost(link,cm,s,costs_all,redirection,expost_intervention=False):
	
	selectline_p = (costs_all.scenarioID==link)&(costs_all.partial_or_full=="partial")
	selectline_f = (costs_all.scenarioID==link)&(costs_all.partial_or_full=="full")
	
	tot = s(cm)*(costs_all.ix[selectline_f,"cost_with_traffic"].values[0]*(steps(cm,redirection))+\
		  costs_all.ix[selectline_p,"cost_with_traffic"].values[0]*(1-steps(cm,redirection)))
		  
	if expost_intervention:
    
		redirect_during_work = 0.1*(1+redirection)
		#additional days off because of reconstruction/rehabilitation
		if s(cm)>15:
			daysoff=30
			# if flood lasts for more than 15 days the road needs to be rebuilt
			tot += costs_all.ix[selectline_f,"construction"].values[0]*costs_all.ix[selectline_f,"KM"].values[0]
		elif (s(cm)<=15)&(s(cm)>7):
			daysoff=10
		else:
			daysoff=2
			
		# the user cost is twice as low as during the flood during road work
		tot += daysoff*costs_all.ix[selectline_f,"KM"].values[0]*\
			((1-redirect_during_work)*costs_all.ix[selectline_p,"cost_with_traffic"].values[0]/2+\
			  redirect_during_work*costs_all.ix[selectline_f,"cost_with_traffic"].values[0])
	return tot
	
def summarize_costs(link,costs_all,climat,redirection,length,expost_intervention=False):
    s = days(length)
    selectline_f = (costs_all.scenarioID==link)&(costs_all.partial_or_full=="full")
    summary = DataFrame(columns=["RP","cost"])
    for rp in [5,10,25,50,100,250,500,1000]:
        cm = 10*costs_all.ix[selectline_f,climat+'_RP'+str(rp)+' (dm)'].values[0]
        summary.loc[len(summary),:]=[rp,totcost(link,cm,s,costs_all,redirection,expost_intervention)]                                      
    return summary
	
def annual_losses_with_interv(link,costs_all,interv,climat,redirection,length,expost_intervention=False):
    if interv=="maintenance":
        length = length/3
        
    summary    = summarize_costs(link,costs_all,climat,redirection,length,expost_intervention)
    new_losses = annual_losses(summary,"cost")
    
    if (interv=="flood_proof"):
        new_losses=0*new_losses
    #elif (interv=="redundancy"):
        #sb = eval("sb_"+link)
    return new_losses
	
def run_scenarios_with_interv(costs_all,link,interv,expost_intervention=False):
    out = DataFrame(columns=['intervention','redirection','climat','length','exp_annual_losses'])

    for redirection in [0,1,2]:
        for climat in ['EU_historical','GFDL_8.5','HadGEM2_8.5','IPSL_8.5']:
            for length in [2,1,0.5]:
                new_losses = annual_losses_with_interv(link,costs_all,interv,climat,redirection,length,expost_intervention)
                out.loc[len(out),:]=[interv,redirection,climat,length,\
                                    new_losses]
    return out
	
def calc_npv(years,costs_all,link,out,interv,disc_rate):

	reduced_losses = run_scenarios_with_interv(link,"bau")-run_scenarios_with_interv(link,interv)

	selectline_f = (costs_all.scenarioID==link)&(costs_all.partial_or_full=="full")

	disc_fact = np.array([1/(1+disc_rate)**i for i in range(1,years+1)])
	disc_reduced_losses = [sum(np.array(years*[l])*disc_fact) for l in reduced_losses]

	if interv=="maintenance":
		interv_cost = costs_all.ix[selectline_f,"maintenance_year"].values[0]*disc_fact*costs_all.ix[selectline_f,"KM"].values[0]*costs_all.ix[selectline_f,"LANES"].values[0]/2
	elif interv=="redundancy":
		sb = eval("sb_"+link)
		interv_cost = sum(sb.upgrade_2_primary*sb.KM*sb.LANES/2)
	else:
		interv_cost = (costs_all.ix[selectline_f,interv] + costs_all.ix[selectline_f,"maintenance_year"].values[0]*disc_fact)*costs_all.ix[selectline_f,"KM"].values[0]*costs_all.ix[selectline_f,"LANES"].values[0]/2
    
	if interv=="redundancy":
		sb = eval("sb_"+link)
		km = sum(sb.KM*sb.LANES/2)
	# else:
		
	npv = disc_reduced_losses - interv_cost
	
	return npv,disc_reduced_losses,interv_cost