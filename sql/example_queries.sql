select
  percentile_disc(0.1) within group (order by percentile_activations.activation),
	percentile_disc(0.2) within group (order by percentile_activations.activation)
from percentile_activations
	group by node_id;


-- insert into percentile_activations(node_id, activation)
-- values
-- (1, 10.5),
-- (1, 11),
-- (1, 12),
-- (2, 5),
-- (2, 6),
-- (2, 7);

select * from percentile_distribution;