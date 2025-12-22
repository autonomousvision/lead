for p in $(grep "path =" .gitmodules | awk '{print $3}'); do
    cd $p
    git pull
    cd - >/dev/null
done

git pull
