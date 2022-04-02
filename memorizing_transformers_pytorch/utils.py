import re
from einops import rearrange

def rearrange_with_dim_list(tensor, pattern, **kwargs):
    regex = r'(\.\.\.[a-zA-Z]+)'
    matches = re.findall(regex, pattern)
    dim_prefixes = tuple(map(lambda t: t.lstrip('...'), set(matches)))

    update_kwargs_dict = dict()

    for prefix in dim_prefixes:
        assert prefix in kwargs, f'dimension list "{prefix}" was not passed in'
        dim_list = kwargs[prefix]
        assert isinstance(dim_list, (list, tuple)), f'dimension list "{prefix}" needs to be a tuple of list of dimensions'
        dim_names = list(map(lambda ind: f'{prefix}{ind}', range(len(dim_list))))
        update_kwargs_dict[prefix] = dict(zip(dim_names, dim_list))

    def sub_with_anonymous_dims(t):
        dim_name_prefix = t.groups()[0].strip('...')
        return ' '.join(update_kwargs_dict[dim_name_prefix].keys())

    pattern_new = re.sub(regex, sub_with_anonymous_dims, pattern)

    for prefix, update_dict in update_kwargs_dict.items():
        del kwargs[prefix]
        kwargs.update(update_dict)

    return rearrange(tensor, pattern_new, **kwargs)
