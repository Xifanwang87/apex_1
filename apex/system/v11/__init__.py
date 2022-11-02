
def load_all_universes():
    """
    0. Load all universe settings from
    /apex.data/apex.system/universes/{universe_name}.yaml
    """
    pass

def init_universe_filesystem(*args, **kwargs):
    """
    
    - folder structure per universe
      {universe_name} /
        settings.yaml
        subuniverse_data / {subuniverse_name} / ## FOLDER STRUCTURE FOR SUBUNIVERSE
        master / {long_only,long_short} / {turnover_level, optimal} /
            portfolio.pq - portfolio        
            returns.pq - return data
            turnover.pq - turnover data
            stats.pq - stats for portfolio


    1. Delete all of /apex.data/apex.universes/{universe_name} 
       if exists and copy {universe_name}.yaml -> /apex/apex.universes/{universe_name}/settings.yaml

    """
    pass


def init_subuniverse_filesystem(*args, **kwargs):
    """
    - folder structure per subuniverse
      {subuniverse_name} /
        input / {market_data.pq, fundamental_data.pq, availability.pq}
        processed / 
            alpha_data /
                {alpha_group.alpha_name} /
                    raw / {base,long_only,long_short}.pq ### The raw data for transforms
                    processed / {long_only,long_short} / {pre,post}.{transform_name}.{turnover_period}.pq
                    master / {long_only,long_short} / {portfolios,returns,stats} / {pre,post}.{transform_name}.{turnover_period}.pq
        master / 
            {long_only,long_short} / 
                {turnover_level, optimal} /
                    portfolio.pq - portfolio        
                    returns.pq - return data
                    turnover.pq - turnover data
                    stats.pq - stats for portfolio

        To combine the portfolios you'll always be parsimonious and use
        1/n. You just need a loss function to tell you when to add the portfolio to the
        group or not.

    """

def load_and_save_data(*args, **kwargs):
    """
     2b. Load subuniverse input data and save to /apex.data/apex.universes/{universe_name}/subuniverse_data/{subuniverse_name}/input
    """
    pass

def compute_alpha_raw(*args, **kwargs):
    pass

def compute_alpha_pre_transforms(*args, **kwargs):
    pass

def compute_alpha_post_transforms(*args, **kwargs):
    pass

def construct_alpha_portfolios(*args, **kwargs):
    pass

def compute_subuniverse_optimal_portfolio(*args, **kwargs):
    pass

def construct_turnover_constrained_subuniverse_portfolios(*args, **kwargs):
    pass

def construct_universe_portfolios(*args, **kwargs):
    pass

# Strategy Logic
def load_all_strategies():
    pass

def init_strategy_filesystem(*args, **kwargs):
    pass

def construct_strategy_subportfolios(*args, **kwargs):
    pass

def combine_strategy_subportfolios(*args, **kwargs):
    pass

def create_strategy_data_output(*args, **kwargs):
    pass

def create_backup_for_strategy_data(*args, **kwargs):
    pass

def create_strategy_emails(*args, **kwargs):
    pass

    