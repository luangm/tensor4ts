import TransformOp from "./TransformOp";

export default class SinOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.sin(a);
  }

}