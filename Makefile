help:
	@echo "test-hdb - Will test a homonegenous dynamic bayesian network"
	@echo "test-fixed-nh-dbn - Will execute a test on a non-homonegenous dynamic baysian network with fixed change-points"
	@echo "test-varying-nh-dbn -  Will run a test on non-homonegenous dynamic bayesian network with varying change-points"
	
test-hdb:
	@echo "Testing a homonegenous-dynamic-bayesian network"
	@(sh ./main_pipeline.sh)

test-varying-nh-dbn:
	@echo "Testing a varying non-homonegenous dynamic bayesian network"
	@(sh ./main_pipeline.sh)

test-varying-nh-dbn:
	@echo "Testing a fixed non-homonegenous dynamic bayesian network"
	@(sh ./main_pipeline.sh)
