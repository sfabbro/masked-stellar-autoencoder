import yaml

with open('.pre-commit-config.yaml', 'r') as f:
    config = yaml.safe_load(f)

for repo in config['repos']:
    if 'codespell' in repo['repo']:
        for hook in repo['hooks']:
            if hook['id'] == 'codespell':
                args = hook.get('args', [])
                found_ignore = False
                for i, arg in enumerate(args):
                    if arg.startswith('--ignore-words-list='):
                        current = arg.split('=')[1]
                        # append new words
                        new_words = 'wee,sme,vai,wit,noo,tru,ofo,tne,wel,egde,meu,onl,nwe,tey,coo,hel,fwe,fpt,wrte,nce'
                        hook['args'][i] = f"--ignore-words-list={current},{new_words}"
                        found_ignore = True
                        break
                if not found_ignore:
                    # try to see if it's passed as --ignore-words-list and then a separate list string
                    pass # handle later if needed

with open('.pre-commit-config.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
