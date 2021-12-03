@0xbfd16a03c247aaa9;



interface Kamodo {


  getFields @0 () -> (fields :List(Field));

    # units='',
    # arg_units=None,
    # data=None,
    # update=None,
    # equation=None,
    # citation=None,
    # hidden_args=[],

  struct Field {
    symbol @0 :Text;
    func @1 :Function;
    defaults @2 :List(Parameter);
    units @3 :Text;
    data @4 :Variable;
    equation @5 :Text;
    citation @6 :Text;
    hidden @7 :List(Text); # Supposed to be for flags, but Function params are of Variable type
  }

  struct Parameter{
    symbol @0 :Text;
    value @1:Variable;
    units @2:Text;
  }


  struct Array{
    data @0 :Data;
    shape @1 :List(UInt32);
    dtype @2 :Text;
  }

  struct GenericVariable(Key, Value){
    symbol @0 :Key;
    value @1 :Value;
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


