"""
RCAEval provides telemetry and ground-truth labels, but NOT the service
dependency graph itself. Since all three systems (Online Boutique, Sock
Shop, Train Ticket) are well-known open-source reference microservice apps,
we can hardcode their architectures here.

IMPORTANT: the Online Boutique graph below is from general knowledge of
Google's "microservices-demo" reference app, not verified against RCAEval's
exact deployed version. Double check this against the RCAEval paper's
system diagrams (or the actual docker-compose / k8s manifests if you can
find them) before trusting it for real experiments -- a wrong topology
would quietly break the topology_consistency evidence check.
"""

# Online Boutique (Google's microservices-demo reference app)
# service -> list of services it directly depends on / calls
ONLINE_BOUTIQUE_TOPOLOGY = {
    "frontend": [
        "productcatalogservice", "cartservice", "currencyservice",
        "checkoutservice", "shippingservice", "recommendationservice",
        "adservice",
    ],
    "checkoutservice": [
        "cartservice", "productcatalogservice", "currencyservice",
        "shippingservice", "paymentservice", "emailservice",
    ],
    "recommendationservice": ["productcatalogservice"],
    "cartservice": [],       # depends on redis, not modeled as a separate service here
    "productcatalogservice": [],
    "currencyservice": [],
    "paymentservice": [],
    "shippingservice": [],
    "emailservice": [],
    "adservice": [],
}

# TODO: fill these in later if/when you use Sock Shop or Train Ticket cases.
# For now, only Online Boutique (RE1-OB) is supported.
SOCK_SHOP_TOPOLOGY = {}
TRAIN_TICKET_TOPOLOGY = {}

TOPOLOGY_BY_SYSTEM = {
    "ob": ONLINE_BOUTIQUE_TOPOLOGY,
    "ss": SOCK_SHOP_TOPOLOGY,
    "tt": TRAIN_TICKET_TOPOLOGY,
}


def get_topology(system_code: str) -> dict:
    """system_code is 're1ob', 're2ss', etc. -- we just need the 2-letter
    system suffix (ob / ss / tt)."""
    suffix = system_code[-2:]
    topo = TOPOLOGY_BY_SYSTEM.get(suffix)
    if not topo:
        raise NotImplementedError(
            f"No topology defined yet for system '{suffix}'. "
            f"Only Online Boutique ('ob') is filled in so far."
        )
    return topo
