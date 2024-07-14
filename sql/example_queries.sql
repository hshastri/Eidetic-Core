select
  percentile_disc(0.1) within group (order by percentile_activations.activation),
	percentile_disc(0.2) within group (order by percentile_activations.activation)
from percentile_activations
	group by node_id;