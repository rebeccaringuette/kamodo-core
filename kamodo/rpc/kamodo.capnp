@0xbfd16a03c247aaa9;



interface Kamodo {


  getFields @0 () -> (fields :List(Field));

  # struct PersonMap {
  #   # Encoded the same as Map(Text, Person).
  #   entries @0 :List(Entry);
  #   struct Entry {
  #     key @0 :Text;
  #     value @1 :Person;
  #   }
  # }

  struct Field {
    symbol @0 :Text;
    func @1 :Function;
    defaults @2 :List(Parameter);
  }

  struct Parameter{
    symbol @0 :Text;
    value @1:Variable;
  }


  # needs to be an interface
  struct Variable{
    data @0 :Data;
    shape @1 :List(UInt32);
    dtype @2 :Text;

  }

  interface Function {
    # A generic function
    # Should use List(Parameter) instead?
    call @0 (params :List(Variable)) -> (result: Variable);
    # Call the function on the given parameters.
  }

}


