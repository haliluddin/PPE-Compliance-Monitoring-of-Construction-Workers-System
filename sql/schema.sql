CREATE TABLE IF NOT EXISTS cameras (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  location TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workers (
  id SERIAL PRIMARY KEY,
  worker_code TEXT UNIQUE,
  registered BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jobs (
  id SERIAL PRIMARY KEY,
  job_type TEXT NOT NULL, -- 'video'|'stream'|'image'
  camera_id INT REFERENCES cameras(id),
  status TEXT NOT NULL DEFAULT 'queued', -- queued|running|completed|failed
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  started_at TIMESTAMP WITH TIME ZONE,
  finished_at TIMESTAMP WITH TIME ZONE,
  meta JSONB
);

CREATE TABLE IF NOT EXISTS violations (
  id SERIAL PRIMARY KEY,
  job_id INT REFERENCES jobs(id) ON DELETE SET NULL,
  camera_id INT REFERENCES cameras(id),
  worker_id INT REFERENCES workers(id) ON DELETE SET NULL,
  worker_code TEXT,
  violation_types TEXT, -- comma or semicolon separated
  frame_index BIGINT,
  frame_ts TIMESTAMP WITH TIME ZONE,
  snapshot BYTEA, -- small jpeg thumbnail
  inference JSONB, -- full detection + keypoints JSON
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_violations_worker_code ON violations(worker_code);
CREATE INDEX IF NOT EXISTS idx_violations_job_id ON violations(job_id);
