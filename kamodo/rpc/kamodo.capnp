@0xbfd16a03c247aaa9;

interface Kamodo {

  getFields @0 () -> (fields :List(Field));

  struct Field {
    symbol @0 :Text;
    func @1 :Function;
  }

  # needs to be an interface
  struct Variable{
    data @0 :Data;
    shape @1 :List(UInt32);
    dtype @2 :Text;

  }

  interface Function {
    # A generic function
    call @0 (params :List(Variable)) -> (result: Variable);
    # Call the function on the given parameters.
  }

}


