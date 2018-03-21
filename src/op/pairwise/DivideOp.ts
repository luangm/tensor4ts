import PairwiseOp from "./PairwiseOp";

export default class DivideOp extends PairwiseOp {

  body(a: number, b?: number): number {
    return a / b;
  }

}