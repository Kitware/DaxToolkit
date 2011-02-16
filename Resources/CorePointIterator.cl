// point_iterator_code
// keywords:
// * index
// * topology_opaque_pointer
// * body
// * input_data_handle
// * output_data_handle
PointIterator iterator_$index$;
point_iterator(&iterator_$index$, $topology_opaque_pointer$);
for (begin_point(&iterator_$index$); !is_done_point(&iterator_$index$);
  next_point(&iterator_$index$))
{
  float input_temp = get_value_point(&iterator_$index$, &$input_data_handle$);
  float output_temp;

  $body$

  set_value_point(&iterator_$index$, &$output_data_handle$, output_temp);
}
