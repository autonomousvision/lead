for p in $(grep "path =" .gitmodules | awk '{print $3}'); do
    cd $p
    git add .
    git commit -m "Update submodule $p"
    git push
    cd - >/dev/null
done

git add .
git commit -m "Update"
git push
