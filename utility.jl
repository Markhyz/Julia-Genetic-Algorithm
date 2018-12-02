macro define(name, body)
  quote
    macro $(esc(name))()
      esc($(Expr(:quote, body)))
    end
  end
end