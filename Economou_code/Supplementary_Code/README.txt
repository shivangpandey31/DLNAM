R code accompanying the paper "A unifying modelling approach for 
hierarchical distributed lag models".

t.economou@cyi.ac.cy

04/Jan/2024

The code is structured in the following folders:

1)	DLNM_GAM_MCMC: File fit_models.R contains R code to fit a 
	GAM-DLNM to the open-source ChicagoNMAPS data using mgcv, 
	and then to also fit the model using full MCMC, using the 
	R package nimble. File results.RData contains the MCMC 
	results. In case computation is too heavy, these can be
	simply loaded at the designated point in the code.
   
2)	SPATIALLY_SMOOTH_DLNM: Contains 1 file, spatially_smooth_DLNMs.R,
	which simulates ficticious data for a situation where the 
	temperature-lag relationship is smoothly varying in space. 
	Uses the ChicagoNMAPS data as the basis for creating the 
	baseline temperature-lag relationship.
	
3)	HIERARCHICAL_STRUCTURES: Simulates data for 5 made-up regions
	using the ChicagoNMAPS data set as the basis. Then fits a
	hierarchical GAM to the simulated data.
	
4)	SINGLE_COVARIATE_MODELS: Contains the following files:
		- DLNMs_as_GAMs.R, which emulates the Poisson model
		fitted to the Thessaloniki data, but it does so for
		ChicagoNMAPS. Includes computation of the forward and
		backward attributable fraction.
		- model_checking.R, which does posterior predictive
		model checking for the Poisson model, then extends to 
		Negative Binomial and then adds to that temporal
		structures. Same analysis as the Thessaloniki data
		but for ChicagoNMAPS.
		- compare_NegBin_with_quasiPoisson.R which fits a
		Negative Binomial and quasiPoisson DLNM as a GAM, and
		then compares (visually) the estimated relaative risk.
		
5)	MULTIPLE_COVARIATE_MODELS: Contains file temp_PM10_interaction.R,
	which fits a DLNM-GAM to quantify the lagged effects from the 
	interaction of temperature and PM10 for ChicagoNMAPS. This exactly
	emulates the analysis in the paper, for the Cyprus data, that looked
	at the interaction between temperature and relative humidity. 
