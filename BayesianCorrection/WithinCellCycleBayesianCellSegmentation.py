from math import log, exp;
from numpy import var;

class WithinCellCycleBayesianCellSegmentation :

    def __init__(self, growthVariance, timeInterval, minGrowthRate, maxGrowthRate):
        self.growthVariance = growthVariance;
        self.timeInterval = timeInterval;
        self.minGrowthRate = minGrowthRate;
        self.maxGrowthRate = maxGrowthRate;
    
    def EstimateLength(self, observedLength, expectedLength, alpha) :
        return [a * o + (1 - a) * e \
            for o, e, a in zip(observedLength, expectedLength, alpha)];
    
    def EstimateGrowthRate(self, cellLength) :
        
        numerator = 0.0;
        denominator = 0.0;
        
        for k in range(1, len(cellLength)) :
            numerator += cellLength[k] * cellLength[k - 1];
            denominator += cellLength[k - 1] * cellLength[k - 1];
    
        if denominator == 0.0 :
            return self.minGrowthRate;
        
        growthRate = max(self.minGrowthRate, log(numerator / denominator) / self.timeInterval);
        growthRate = min(self.maxGrowthRate, growthRate);
        return growthRate;
    
    def ExpectedLength(self, cellLength, growthRate) :
    
        expectedLength = [0] * len(cellLength);
        
        expectedLength[0] = cellLength[0];
        for k in range(1, len(cellLength)) :
            expectedLength[k] = cellLength[k - 1] * exp(growthRate * self.timeInterval);
        return expectedLength;
        
    def EstimateAlpha(self, observedLength, inferredLength) :
        return [self.growthVariance / ((o - i) ** 2 + self.growthVariance) \
            for o,i in zip(observedLength, inferredLength)];
    
    def Inference(self, observedLength, iteration) :

        growthRate = 0;
        expectedLength = [0] * len(observedLength);
        alpha = [0] * len(observedLength);
        inferredLength = [0] * len(observedLength);
        
        for i in range(iteration) :
 
            print('-------------------------------------------------');
            print('observed length:');
            print(observedLength);
            
            growthRate = self.EstimateGrowthRate(inferredLength);
            print("growth rate: ", growthRate);
            
            expectedLength = self.ExpectedLength(inferredLength, growthRate);
            print("expected length: ");
            print(expectedLength);
            
            alpha = self.EstimateAlpha(observedLength, inferredLength);
            print("alpha");
            print(alpha);
            
            inferredLength = self.EstimateLength(observedLength, expectedLength, alpha);

            print('inferred length:');
            print(inferredLength);

        return inferredLength;

