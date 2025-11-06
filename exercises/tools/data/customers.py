DATASOURCE = {
    1: {
        'name': 'Acme Lightbulbs',
        'annual_spend': 923468,
        'offices_in': ['US', 'GB', 'DK']
        },
    2: {
        'name': 'Bargain Basements',
        'annual_spend': 4578,
        'offices_in': ['US', 'FR', 'ZA', 'SE']
        },
    3: {
        'name': 'Nordic Ventures',
        'annual_spend': 1350000,
        'offices_in': ['SE', 'NO', 'DK']
        },
    4: {
        'name': 'Pacific Techware',
        'annual_spend': 245000,
        'offices_in': ['US', 'CA', 'AU']
        },
    5: {
        'name': 'EuroMart Wholesale',
        'annual_spend': 78543,
        'offices_in': ['DE', 'NL', 'BE']
        },
    6: {
        'name': 'Baltic Logistics',
        'annual_spend': 312000,
        'offices_in': ['LT', 'LV', 'EE', 'SE']
        },
    7: {
        'name': 'Aurora Foods',
        'annual_spend': 1689000,
        'offices_in': ['NO', 'SE']
        },
    8: {
        'name': 'Sapphire Systems',
        'annual_spend': 501200,
        'offices_in': ['GB', 'IE']
        },
    9: {
        'name': 'Maple Outfitters',
        'annual_spend': 91000,
        'offices_in': ['CA']
        },
    10: {
        'name': 'Andes Apparel',
        'annual_spend': 120450,
        'offices_in': ['AR', 'CL', 'PE']
        }
}

def get_customer_list():
    return [{"id": id, "name": customer['name']} for id, customer in DATASOURCE.items()]

def get_customer_annual_spend(id):
    return DATASOURCE[id]['annual_spend']

def get_customer_office_locations(id):
    return DATASOURCE[id]['offices_in']
