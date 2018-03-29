import TransformOp from "./TransformOp";

export default class CeilOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.ceil(a);
  }

}