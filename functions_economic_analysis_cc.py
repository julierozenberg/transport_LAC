import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from pandas import DataFrame

from scipy import interp


def days(length):
	'''length of disruption in days as a function of water depth in cm''' 
	s = InterpolatedUnivariateSpline([0, 5, 30, 50, 200, 300], [0,1*length,5*length,10*length,30*length,50*length], k=1)
	return s

def steps(cm,redirection):
	''' share of traffic redirected to the second best route as a function of water depth in cm (3 scenarios)''' 
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
	Calculates expected annual losses based on losses per return period event. it takes as an input a dataframe that as a 'RP' column. the second argument is the name of the column that contains the losses for each RP
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

def tot_huaycos(link,costs_all,duration_huaycos,improved_2nd,expost_intervention=False):
	''' 
	for landslide we assume there is a full disruption for all events. this function calculates the user losses + reconstruction losses
	''' 
	selectline_f = (costs_all.scenarioID==link)&(costs_all.improved_2nd==improved_2nd)&(costs_all.partial_or_full=="full")
	tot = duration_huaycos*costs_all.ix[selectline_f,"cost_with_traffic"].values[0]
	if expost_intervention:
		daysoff = 30
		tot += costs_all.ix[selectline_f,"construction"].values[0]*costs_all.ix[selectline_f,"KM"].values[0]
		tot += daysoff*costs_all.ix[selectline_f,"cost_with_traffic"].values[0]
		
	return tot

def usercost(link,days,costs_all,share_traffic,improved_2nd):
	''' user cost for a disruption event -- can take into account the fact that the second best is improved (in that case RUC increase is lower)'''

	selectline_p = (costs_all.scenarioID==link)&(costs_all.improved_2nd==improved_2nd)&(costs_all.partial_or_full=="partial")
	selectline_f = (costs_all.scenarioID==link)&(costs_all.improved_2nd==improved_2nd)&(costs_all.partial_or_full=="full")

	tot = days*(costs_all.ix[selectline_f,"cost_with_traffic"].values[0]*(share_traffic)+\
		  costs_all.ix[selectline_p,"cost_with_traffic"].values[0]*(1-share_traffic))

	if improved_2nd:
		redirect_during_work=1
	else:
		redirect_during_work = 0.5
	#additional days off because of reconstruction/rehabilitation
	if share_traffic==1:
		#if huaycos
		daysoff=30
	else:
		if days>15:
			daysoff=30
		elif (days<=15)&(days>7):
			daysoff=10
		else:
			daysoff=2
	# the user cost is twice lower during the flood than during road work
	tot += daysoff*((1-redirect_during_work)*costs_all.ix[selectline_p,"cost_with_traffic"].values[0]/2+redirect_during_work*costs_all.ix[selectline_f,"cost_with_traffic"].values[0])
	return tot

def intervention_cost(costs_all,link,improved_2nd,typeofinterv):
	'''cost of intervention*km ''' 
	selectline_f = (costs_all.scenarioID==link)&(costs_all.improved_2nd==improved_2nd)&(costs_all.partial_or_full=="full")
	tot = costs_all.ix[selectline_f,typeofinterv].values[0]*costs_all.ix[selectline_f,"KM"].values[0]
	return tot
		

def get_water_level(costs_all,link,proba,climat):
	'''extracts water levels. the 'proba' argument is how much frequency is increased because of climate change or el nino''' 
	water = DataFrame(columns=['return_period','water_level','proba'])
	for RP in [5,10,25,50,100,250,500,1000]:
		col = "{}_RP{} (dm)".format(climat,RP)
		water.loc[len(water),:]=[RP/proba,costs_all.ix[(costs_all.scenarioID==str(link))&(costs_all.partial_or_full=="full")&(costs_all.improved_2nd==0),col].values[0],proba]
	inter = water.copy()
	#s = InterpolatedUnivariateSpline(water['return period'], water['water level'],k=1)
	water.loc[len(water),:] = [500,interp([500],inter['return_period'].astype(float), inter['water_level'].astype(float))[0],proba]
	water.loc[len(water),:] = [1000,interp([1000],inter['return_period'].astype(float), inter['water_level'].astype(float))[0],proba]
		
	return water

def summarize_costs_floods(link,costs_all,climat,redirection,length,proba,exanteinterv):
	summary = DataFrame(columns=["RP","user cost","ex-post interv"])
	water = get_water_level(costs_all,link,proba,climat)

	if exanteinterv=="flood_proof":
		for i in water.index:
			rp = water.loc[i,"return_period"]
			summary.loc[len(summary),:]=[rp,0,0]
	else:
		if exanteinterv=="bau":
			improved_2nd=0
		elif exanteinterv=="maintenance":
			length = length/3
			improved_2nd=0
		elif exanteinterv=="redundancy":
			improved_2nd=1

		s = days(length)
		for i in water.index:
			rp = water.loc[i,"return_period"]
			cm = 10*water.loc[i,"water_level"]
			disruption_length = s(cm)
			share_traffic = steps(cm,redirection)
			if s(cm)>15:
				expost = intervention_cost(costs_all,link,improved_2nd,"construction")
			else:
				expost = intervention_cost(costs_all,link,improved_2nd,"rehabilitation")
			summary.loc[len(summary),:]=[rp,usercost(link,disruption_length,costs_all,share_traffic,improved_2nd),expost]
	return summary

def costs_huaycos(link,costs_all,duration_huaycos,exanteinterv):
	''' for landslides''' 
	summary = DataFrame(columns=["user cost","ex-post interv"])
	if exanteinterv=='tunnel':
		summary.loc[0,:]=[0,0]
	else:
		if (exanteinterv=='bau')|(exanteinterv=='maintenance'):
			improved_2nd = 0
		elif (exanteinterv=="redundancy"):
			improved_2nd=1
		summary = DataFrame(columns=["user cost","ex-post interv"])
		share_traffic=1
		summary.loc[0,:]=[usercost(link,duration_huaycos,costs_all,share_traffic,improved_2nd),intervention_cost(costs_all,link,improved_2nd,"construction")]                         
	return summary

def annual_losses_floods(summary):
	user_losses = annual_losses(summary,"user cost")
	expost_losses = annual_losses(summary,"ex-post interv")

	all_losses = [user_losses,expost_losses]
	return all_losses

def run_scenarios_huaycos(costs_all,link,h_proba,exanteinterv,duration_huaycos):

	out = DataFrame(columns=['intervention','duration_huaycos','user_annual_losses','expost_annual_losses'])
	new_losses = h_proba*costs_huaycos(link,costs_all,duration_huaycos,exanteinterv).values[0]
	out.loc[len(out),:]=[exanteinterv,duration_huaycos]+list(new_losses)
	return out

def run_scenarios_floods(costs_all,link,exanteinterv):
	out = DataFrame(columns=['intervention','redirection','proba','length','user_annual_losses','expost_annual_losses'])
	climat = 'EU_historical'
	for climat in ['EU_historical','GFDL_8.5','HadGEM2_8.5','IPSL_8.5']:
		for redirection in [0,1,2]:
			for length in [2,1,0.5]:
				for proba in [1,2]:
					summary = summarize_costs_floods(link,costs_all,climat,redirection,length,proba,exanteinterv)
					out.loc[len(out),:]=[exanteinterv,redirection,proba,length]+annual_losses_floods(summary)
	return out

def run_scenarios_huaycos_from_floods(out_flood,costs_all,link,exanteinterv):
	for i in out_flood.index:
		[exanteinterv,redirection,proba,length] = out_flood.loc[i,['intervention','redirection','proba','length']]
		duration_huaycos = length*10
		h_losses = proba*costs_huaycos(link,costs_all,duration_huaycos,exanteinterv).values[0]
		out_flood.loc[i,"losses_user_huyacos"]  = h_losses[0]
		out_flood.loc[i,"losses_exposti_huycas"]  = h_losses[1]
	return out_flood

def calc_npv_floods(years,costs_all,link,exanteinterv,disc_rate,sb,growth=0,pc_improvement=1,plushuyacos=False):

	baulosses = run_scenarios_floods(costs_all,link,"bau")
	intervlosses = run_scenarios_floods(costs_all,link,exanteinterv)

	reduced_losses = baulosses.user_annual_losses+baulosses.expost_annual_losses-intervlosses.user_annual_losses*pc_improvement-intervlosses.expost_annual_losses

	if plushuyacos:
		baulosses = run_scenarios_huaycos_from_floods(baulosses,costs_all,link,"bau")
		intervlosses = run_scenarios_huaycos_from_floods(intervlosses,costs_all,link,exanteinterv)
		reduced_losses = reduced_losses+baulosses.losses_user_huyacos+baulosses.losses_exposti_huycas-intervlosses.losses_user_huyacos*pc_improvement-intervlosses.losses_exposti_huycas

	out = DataFrame(columns=['redirection','proba','length','npv','disc_reduced_losses','interv_cost'],index=baulosses.index)
	out[['intervention','redirection','proba','length']]=intervlosses[['intervention','redirection','proba','length']]

	selectline_f = (costs_all.scenarioID==link)&(costs_all.partial_or_full=="full")&(costs_all.improved_2nd==0)

	disc_fact = np.array([1/(1+disc_rate)**i for i in range(1,years+1)])
	growth = np.array([(1+growth)**i for i in range(1,years+1)])
	out.loc[:,'disc_reduced_losses'] = [sum(np.array(years*[l])*disc_fact*growth) for l in reduced_losses]

	if exanteinterv=="maintenance":
		out.loc[:,'interv_cost'] = sum(costs_all.ix[selectline_f,"maintenance_year"].values[0]*disc_fact*costs_all.ix[selectline_f,"KM"].values[0]*costs_all.ix[selectline_f,"LANES"].values[0]/2)
	elif exanteinterv=="redundancy":
		remove = (sb.CLASS=="Primary")&(sb.SURF=="Paved")&(sb.CONDITION=="Bueno")
		sbtemp = sb.ix[~remove,:].copy()
		out.loc[:,'interv_cost'] = sum(sbtemp.upgrade_2_primary*sbtemp.KM*sbtemp.LANES/2*pc_improvement)
	elif exanteinterv=="flood_proof":
		out.loc[:,'interv_cost'] = (costs_all.ix[selectline_f,"flood_proof_1m"].values[0] + sum(costs_all.ix[selectline_f,"maintenance_year"].values[0]*disc_fact))*costs_all.ix[selectline_f,"KM"].values[0]*costs_all.ix[selectline_f,"LANES"].values[0]/2
	out.loc[:,'npv'] = out.disc_reduced_losses-out.interv_cost

	return out

def calc_npv_huaycos(years,costs_all,link,expostinterv,disc_rate,h_proba,sb,growth,duration_huaycos):

	baulosses = run_scenarios_huaycos(costs_all,link,h_proba,"bau",duration_huaycos)
	intervlosses = run_scenarios_huaycos(costs_all,link,h_proba,expostinterv,duration_huaycos)

	out = DataFrame(columns=['intervention','duration_huaycos','npv','disc_reduced_losses','interv_cost'],index=baulosses.index)
	out.duration_huaycos=baulosses.duration_huaycos
	out.intervention=intervlosses.intervention

	reduced_losses = baulosses.user_annual_losses+baulosses.expost_annual_losses-intervlosses.user_annual_losses-intervlosses.expost_annual_losses

	selectline_f = (costs_all.scenarioID==link)&(costs_all.partial_or_full=="full")&(costs_all.improved_2nd==0)

	disc_fact = np.array([1/(1+disc_rate)**i for i in range(1,years+1)])
	growth = np.array([(1+growth)**i for i in range(1,years+1)])
	out.loc[:,'disc_reduced_losses'] = [sum(np.array(years*[l])*disc_fact*growth) for l in reduced_losses]

	if expostinterv=="redundancy":
		remove = (sb.CLASS=="Primary")&(sb.SURF=="Paved")&(sb.CONDITION=="Bueno")
		sbtemp = sb.ix[~remove,:].copy()
		out.loc[:,'interv_cost'] = sum(sbtemp.upgrade_2_primary*sbtemp.KM*sbtemp.LANES/2)
	elif expostinterv=="tunnel":
		out.loc[:,'interv_cost'] = (costs_all.ix[selectline_f,"tunnel"].values[0])*costs_all.ix[selectline_f,"KM"].values[0]

	out.loc[:,'npv'] = out.disc_reduced_losses-out.interv_cost

	return out