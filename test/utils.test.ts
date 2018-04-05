import ShapeUtils from "../src/utils/ShapeUtils";

test("broadcast shapes", function () {
  let a: number[] = [];
  let b: number[] = [17];

  let z = ShapeUtils.broadcastShapes(a, b);
  console.log(z);

  let s = ShapeUtils.getReductionIndices(a, b);
  console.log(s);

  expect(s.left).toEqual([0]);
});