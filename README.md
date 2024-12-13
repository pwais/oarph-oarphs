Repo for blog at [https://pwais.github.io/oarph-oarphs/](https://pwais.github.io/oarph-oarphs/).

## Dev

```
docker run --rm --net=host -it -v `pwd`:/opt/oo:z -w /opt/oo jekyll/jekyll:3.8.6 bash
bundle install
bundle exec jekyll serve --port 4434 --host "ur.ip"
```

To update jekyll, delete the `Gemfile.lock` and run (in docker) `bundle update`.

