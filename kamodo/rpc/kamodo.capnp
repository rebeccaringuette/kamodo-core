@0xbfd16a03c247aaa9;


interface Kamodo {
  struct Map(Key, Value) {
    entries @0 :List(Entry);
    struct Entry {
      key @0 :Key;
      value @1 :Value;
    }
  }

  getFields @0 () -> (fields :Map(Text, Field));


  # everything needed for registration
  struct Field {
    func @0 :Function;
    meta @1 :Meta;
    data @2 :Variable;
  }

  # match kamodo's meta attribute
  struct Meta {
    units @0 :Text;
    argUnits @1 :Map(Text, Text);
    citation @2 :Text;
    equation @3 :Text; # latex expression
    hiddenArgs @4 :List(Text);
  }

  struct Argument {
    name @0 :Text;
    value @1 :Variable;
  }

  # needs to be an interface
  struct Variable {
    data @0 :Data;
    shape @1 :List(UInt32);
    dtype @2 :Text;

  }

  interface Function {
    # A pythonic function f(*args, **kwargs)
    call @0 (args :List(Variable), kwargs :List(Argument)) -> (result: Variable);
    getArgs @1 () -> (args :List(Text));
    getKwargs @2 () -> (kwargs: List(Argument));
  }

}


