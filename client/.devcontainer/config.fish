if test "$PAGER" = "head -n 10000 | cat" -o "$COMPOSER_NO_INTERACTION" = "1"
    exit
end

if status is-interactive; and test "$TERM" != "dumb"
    # Commands to run in interactive sessions can go here
end
