create table percentile_activations(
	node_id		int not null,
	activation	double precision not null
);

CREATE INDEX node_idx ON percentile_activations (node_id);