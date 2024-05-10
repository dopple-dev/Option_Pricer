class Par:

    def __init__(self, S0=0, K=0, T=1, v0=0, sigma0=0, payoff='call', exercise_style='European'):
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.v0 = v0
        self.sigma0 = sigma0

        if (payoff == 'call' or payoff == 'c' or payoff == 'Call' or payoff == 'C' or 
            payoff == 'put' or payoff == 'p' or payoff == 'Put' or payoff == 'P'):
            self.payoff = payoff 
        else:
            raise ValueError('invalid option payoff type.')

        
        if exercise_style == 'European':
            self.exercise_style = exercise_style
        else: 
            raise ValueError('invalid exercise_style type. additional implementation needed for ', exercise_style)