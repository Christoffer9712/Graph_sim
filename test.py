from utils.utils import elevation_angle_deg, load_europe_land_shape
from ground_network.gateways    import get_gateways
from ground_network.upf        import get_upfs
from ground_network.aviation     import generate_aviation_nodes
from ground_network.network      import build_gateway_graph, build_aviation_graph, build_upf_graph  
from ground_network.topology     import build_static_ground_graph
from e2e_network.topology import build_full_graph

angle = elevation_angle_deg(
    ground_lat = 52.0, ground_lon = 13.0,
    sat_lat = 52.0, sat_lon = 13.0,
    sat_alt_m = 550000.0
)

print(f"Elevation angle: {angle:.2f} degrees") # should be 90 degrees for a satellite directly overhead at 550 km altitude

gateways = get_gateways()
print(f"Loaded {len(gateways)} gateways:")
for gw in gateways[:5]:  # print first 5 gateways
    print(f"  {gw.node_id}: type={gw.node_type.name}, lat={gw.lat:.2f}, lon={gw.lon:.2f}")

aviation_nodes = generate_aviation_nodes(n_target=300)
print(f"Generated {len(aviation_nodes)} aviation nodes:")
for av in aviation_nodes[:5]:  # print first 5 aviation nodes
    print(f"  {av.node_id}: type={av.node_type.name}, lat={av.lat:.2f}, lon={av.lon:.2f}")
    

gw_graph = build_gateway_graph(gateways)
print(f"Gateway graph has {gw_graph.number_of_nodes()} nodes and {gw_graph.number_of_edges()} edges.")

av_graph = build_aviation_graph(aviation_nodes)
print(f"Aviation graph has {av_graph.number_of_nodes()} nodes and {av_graph.number_of_edges()} edges.")

upf_graph = build_upf_graph(get_upfs())
print(f"UPF graph has {upf_graph.number_of_nodes()} nodes and {upf_graph.number_of_edges()} edges.")    

static_ground = build_static_ground_graph(gw_graph, av_graph, upf_graph)
print(f"Static ground graph has {static_ground.number_of_nodes()} nodes and {static_ground.number_of_edges()} edges.")