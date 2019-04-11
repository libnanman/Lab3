import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = regentlib.c

fspace Page {
  rank         : double,
  next_rank    : double,
  numLinks_out : double,
  numLinks_in  : double,
  color        : uint8
}

fspace Link(p : region(Page), p2 : region(Page)) {
  dest         : ptr(Page, p),
  src          : ptr(Page, p2),
  color        : int1d
}


terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
                      r_links   : region(Link(r_pages, r_pages)),
                      damp      : double,
                      num_pages : uint64,
                      filename  : int8[512],
                      parall    : uint8)
where
  reads writes(r_pages, r_links)
do
  var ts_start = c.legion_get_current_time_in_micros()
  var total_links : uint64
  total_links = 0
  var num : uint64
  num = 0
  var pages_per : uint64
  pages_per = num_pages/parall
  pages_per += 1
  var current_color : uint8
  current_color = 0
  for page in r_pages do
    page.rank = 1.0 / num_pages
    page.next_rank = 0
    page.numLinks_out = 0
    page.numLinks_in = 0
    page.color = current_color % parall
    num += 1
    if num > pages_per then
      num = 0
      current_color += 1
    end
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = dynamic_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = dynamic_cast(ptr(Page, r_pages), page_ids[1])
    src_page.numLinks_out += 1
    dst_page.numLinks_in += 1
    total_links += 1
    link.dest = dst_page
    link.src = src_page
    --link.color = num % parall
    --num += 1
  end

  var links_per : uint64
  links_per = total_links / parall
  links_per += 1
  var current_link : uint64[32]
  for i=0,31 do
    current_link[i] = 0
  end
 
--  var rand : uint64
--  rand = 1

  for link in r_links do
    if current_link[link.dest.color] > links_per then
      link.dest.color = (link.dest.color + 1) % parall
      link.color = link.dest.color
      current_link[link.dest.color] += 1
    else
      link.color = link.dest.color
      current_link[link.dest.color] += 1
    end
  end


  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end


task compute_ranks(
                   r_pages_dest : region(Page),
                   r_pages_src  : region(Page),
                   r_links      : region(Link(r_pages_dest, r_pages_src)),
                   damp         : double)

where reads (r_links, r_pages_src.rank, r_pages_src.numLinks_out), reduces+(r_pages_dest.next_rank) do

  --var temp_rand = (1-damp)/num_pages

  for link in r_links do
    link.dest.next_rank += damp*(link.src.rank/link.src.numLinks_out)-- + (temp_rand/link.dest.numLinks_in)
  end

  --for page in r_pages_dest do
  --  page.next_rank += temp_rand
  --end

end


task compute_err_serial (r_pages     : region(Page),
                         error_bound : double,
                         damp        : double,
                         num_pages   : uint64)

where reads writes(r_pages) do

  var err_total : double
  err_total = 0

  var temp_rand = (1-damp)/num_pages

  for page in r_pages do
    page.next_rank += temp_rand
    err_total += (page.next_rank-page.rank)*(page.next_rank-page.rank)
    page.rank = page.next_rank
    page.next_rank = 0
  end

  if err_total > error_bound*error_bound then
    return false
  end
  return true

end


task dump_ranks(r_pages  : region(Page),
                filename : int8[512])
where
  reads(r_pages.rank)
do
  var f = c.fopen(filename, "w")
  for page in r_pages do c.fprintf(f, "%g\n", page.rank) end
  c.fclose(f)
end

task toplevel()
  var config : PageRankConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Pages  : %11lu *\n",  config.num_pages)
  c.printf("* Number of Links  : %11lu *\n",  config.num_links)
  c.printf("* Damping Factor   : %11.4f *\n", config.damp)
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("* # Parallel Tasks : %11u *\n",   config.parallelism)
  c.printf("**********************************\n")

  -- Create a region of pages
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  var r_links = region(ispace(ptr, config.num_links), Link(wild, wild))
--  var parl_space = ispace(int1d, config.parallelism)

  var c_space = ispace(int1d, config.parallelism)
--  var r_err = region(c_space, Err)

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input, config.parallelism)


  var p_links       = partition(r_links.color, c_space)
  var p_pages_src   = image(r_pages, p_links, r_links.src)
  var p_pages_dest  = image(r_pages, p_links, r_links.dest)

  var num_iterations = 0
  var converged = false
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_start = c.legion_get_current_time_in_micros()
  while not converged do
    num_iterations += 1
    __demand(__parallel)
    for c in c_space do
      compute_ranks(p_pages_dest[c], p_pages_src[c], p_links[c], config.damp)
    end
    converged = compute_err_serial(r_pages, config.error_bound, config.damp, config.num_pages)
  end
  __fence(__execution, __block) -- This blocks to make sure we only time the pagerank computation
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)
