import dogpile.cache as dc

DATASTORE_CACHE = dc.make_region(key_mangler=lambda key: "apex:apex_data:v1.12:" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 2,
        'redis_expiration_time': 60*60*24,   # 24h
    },
)

METADATA_DB_CACHE = dc.make_region(key_mangler=lambda key: "apex:security.metadata.1" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 3,
        'redis_expiration_time': 60*60*24*30,   # 30 days
    },
)

PORTFOLIO_DB_CACHE = dc.make_region(key_mangler=lambda key: "apex:portfolio_db:account_portfolios:bbg_pull:" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 4,
        'redis_expiration_time': 60*60*24,   # day
    },
)

PORTFOLIO_DB_SHORT_TERM_CACHE = dc.make_region(key_mangler=lambda key: "apex:portfolio_db:account_portfolios:bbg_pull:" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 5,
        'redis_expiration_time': 60*60*6,   # 6 hours
    },
)

INDEX_WEIGHTS_CACHING = dc.make_region(key_mangler=lambda key: "apex:index:weights:v2.0b" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 6,
        'redis_expiration_time': 60*60*24,   # A day
    },
)

MARKET_DATA_CACHING = dc.make_region(key_mangler=lambda key: "apex:market_data:caching:1.1" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 7,
        'redis_expiration_time': 60*60*8,   # 8 hours
    },
)

FUNDAMENTAL_DATA_CACHING = dc.make_region(key_mangler=lambda key: "apex:fundamental_data:caching:1.1" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 9,
        'redis_expiration_time': 60*60*8,   # 8 hours
    },
)

UNIVERSE_DATA_CACHING = dc.make_region(key_mangler=lambda key: "apex:universe:caching:1.1" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 8,
        'redis_expiration_time': 60*60*8,   # 8 hours
    },
)
