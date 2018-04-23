import PairwiseOp from "./PairwiseOp";
import Tensor from "../../Tensor";

export default class AddOp extends PairwiseOp {

  constructor(x: Tensor, y: Tensor, z: Tensor) {
    super(x, y, z);
  }

  body(a: number, b: number): number {
    return a + b;
  }

}