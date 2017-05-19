cluster_spec = dict()
cluster_spec['ps'] = list()
cluster_spec['cn'] = list()

# parameter server
cluster_spec['ps'].append({'IP':'127.0.0.1', 'Port':8888})

# computing node
cluster_spec['cn'].append({'IP':'127.0.0.1','Port':60000})
cluster_spec['cn'].append({'IP':'127.0.0.1','Port':60001})
