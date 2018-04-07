function loadRealsense(file_name, root_path)
  print(string.format("Loading data from %s", file_name))
  local output = {}
  local file = io.open(file_name, "r")
  for line in file:lines() do

    local data = {}

    baseline = root_path .. line
    data.color = baseline .. '_color.png'
    data.depth = baseline .. '_depth.png'

    data.name = baseline

    table.insert(output, data)
  end
  print(string.format("%d data loaded.",#output))
  return output
end

function loadScanNetRender(file_name, root_path)
  print(string.format("Loading data from %s", file_name))
  local output = {}
  local file = io.open(file_name, "r")
  for line in file:lines() do
    line = root_path .. line 

    local data = {}

    baseline = line
    data.color = baseline:gsub('data_dir', 'color'):gsub('_suffix', '.jpg');

    baseline = line
    data.nx = baseline:gsub('data_dir', 'mesh_images'):gsub('_suffix', '_mesh_nx.png');
    data.ny = baseline:gsub('data_dir', 'mesh_images'):gsub('_suffix', '_mesh_ny.png');
    data.nz = baseline:gsub('data_dir', 'mesh_images'):gsub('_suffix', '_mesh_nz.png');

    baseline = line
    data.depth = baseline:gsub('data_dir', 'mesh_images'):gsub('_suffix', '_mesh_depth.png');

    baseline = line
    data.depth_raw = baseline:gsub('data_dir', 'depth'):gsub('_suffix', '.png');

    data.name = line

    table.insert(output, data)
  end
  print(string.format("%d data loaded.",#output))
  return output
end


