create database eidetic;

create table percentile_activations(
	node_id		int not null,
	activation	double precision not null
);

CREATE INDEX node_idx ON percentile_activations (node_id);


create table percentile_distribution(
	node_id		int not null,
	threshold_1	double precision not null,
	threshold_2 double precision not null
);

CREATE INDEX node_idx_dist ON percentile_distribution (node_id);