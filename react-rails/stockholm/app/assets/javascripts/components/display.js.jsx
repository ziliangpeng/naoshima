class Display extends React.Component {
  render() {
    var images = this.props.images.map(function (image) {
      return (
        <div>
          <p>{image.url}</p>
          <img src={image.url} height="100"></img>
        </div>
      );
    });

    return (
      <div>
        {images}
      </div>
    );
  }
}
