import PairwiseOp from "./PairwiseOp";

export default class MultiplyOp extends PairwiseOp {

  body(a: number, b?: number): number {
    return a * b;
  }

}